import functools
import time

import numpy as np
import torch

import src.minterpy.utils as utils
from src.minterpy.solvers import Interpolator

TIMING = True
TIMES = {}


def timer(func):
    """simple timing decorator"""
    if TIMING:
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start = time.time()  # 1
            value = func(*args, **kwargs)
            run_time = time.time() - start
            TIMES[func.__name__] = run_time
            return value
    else:
        wrapper_timer = func
    return wrapper_timer


class transform(Interpolator):

    def buildCanonicalVandermonde(self, X, gamma):
        """
        Canonical exponents for polynomial interpolation in normal form
        Y = a * V
        a.. coefficients
        V.. vandermonde exponents
        Y.. interpolated signal
        Example:
            x_in: input vector
            Y_hat = torch.mm(canon_coefs,
                    buildCanonicalVandermonde(x_in, gamma).t())
        
        Parameters:
        X.. X positions
        gamma.. gamma vectors of our polynomial
        """
        noX, dimX = X.shape
        dimGamma, noCoefficients = gamma.shape
        assert dimX == dimGamma, "Input dimensions (%d,%d) of (X, gamma) dont match." % (dimX, dimGamma)
        V = torch.ones((noX, noCoefficients))
        # TODO: this probably only works for m == 2
        for j in range(noCoefficients):
            for k in range(dimX):
                V[:, j] *= (X[:, k] ** gamma[k, j])
            # V[:,j] = (X[:,0]**gamma[0,j]) * (X[:,1]**gamma[1,j])
        return V

    def getGammaDXi(self, i=0):
        """
        This function returns the monomials for the first derivative in x_i direction.
        Constants are scaling factors for the coefficients.
        Example:
            x_in: input vector
            # compute monimials for derivative in x_0 direction
            gamma_dx0, constants_dx0 = getGammaDXi(i = 0)
            # predict values
            Y_hat_dx0 = torch.mm( (canon_coefs*constants_dx0).float(), 
                                  buildCanonicalVandermonde(x_in, gamma_dx0).t())
        """
        gamma2_dxi = self.trans_gamma.clone()
        gamma2_dxi[i, :] -= 1
        voidCoefs = (gamma2_dxi[i, :] == -1)
        gamma2_dxi[:, voidCoefs] = -1

        constants = self.trans_gamma[i, :]

        return gamma2_dxi, constants

    @timer
    def __init__(self, m=2, n=2, lp_degree=2):
        Interpolator.__init__(self, m, n, lp_degree)
        self.__build_vandermonde_n2c()
        self.__build_transform_n2c()
        self.__build_vandermonde_l2n()
        self.__build_transform_l2n()
        self.__build_trans_Matrix()

    @timer
    def __build_vandermonde_n2c(self):
        """
        Vandermonde Matrix for Newton to canonical transformation
        :return:
        """
        self.init_gamma = np.zeros((self.m, 1))
        self.trans_gamma = torch.from_numpy(
            utils.Gamma_lp(self.m, self.n, self.init_gamma, self.init_gamma.copy(), self.lp_degree))
        self.V_n2c = torch.ones((self.N, self.N), dtype=torch.float32)
        for i in range(0, self.N):
            for j in range(1, self.N):
                for d in range(0, self.m):
                    self.V_n2c[i, j] = self.V_n2c[i, j] * self.tree.grid_points[d, i] ** self.trans_gamma[d, j]

    @timer
    def __build_transform_n2c(self):
        """
        Transformation from Newton to canonical form
        :return:
        """
        self.Cmn_n2c = torch.zeros((self.N, self.N), dtype=torch.float32)
        for j in range(self.N):
            self.Cmn_n2c[:, j] = torch.from_numpy(
                self.run(self.m, self.N, self.tree.tree, self.V_n2c[:, j].numpy(), self.tree.grid_values.copy(),
                         self.gamma_placeholder.copy(), 1, 1))
        # TODO: Replace LU-based solver by one exploiting triangular structures
        # self.inv_Cmn_n2c =  torch.from_numpy(solve_triangular(self.Cmn_n2c,torch.eye(self.N)))
        self.inv_Cmn_n2c = torch.solve(torch.eye(self.N, dtype=torch.float32), self.Cmn_n2c).solution

    @timer
    @timer
    def __build_vandermonde_l2n(self):
        """
        Vandermonde Matrix for Lagrange to Newton transformation
        :return:
        """
        self.V_l2n = torch.eye(self.V_n2c.shape[0], dtype=torch.float32)

    @timer
    def __build_transform_l2n(self):
        """
        Lagrange to Newton transformation
        :return:
        """
        self.Cmn_l2n = torch.zeros((self.N, self.N), dtype=torch.float32)
        for j in range(self.N):
            self.Cmn_l2n[:, j] = torch.from_numpy(
                self.run(self.m, self.N, self.tree.tree, self.V_l2n[:, j].numpy().copy(), self.tree.grid_values.copy(),
                         self.gamma_placeholder.copy(), 1, 1))

    @timer
    def transform_l2n(self, l):
        return torch.mm(self.Cmn_l2n, l.view(-1, 1))

    def transform_l2c(self, v):
        return torch.mm(self.trans_matrix, v.view(-1, 1))

    def transform_n2c(self, d):
        return torch.mm(self.inv_Cmn_n2c, d.view(-1, 1))

    @timer
    def __build_trans_Matrix(self):
        self.trans_matrix = torch.mm(self.inv_Cmn_n2c, self.Cmn_l2n).float()
