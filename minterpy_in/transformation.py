# -*- coding:utf-8 -*-
import numpy as np
import scipy
from scipy.linalg import solve_triangular

from minterpy.diagnostics import TIMES, timer
from minterpy.solvers import Interpolator


# TODO again child class. hierarchy really needed?
# TODO rather define "interpolant" class as polynomial.
#  then lazily evaluate different coefficients (<- transform to basis) on demand
# TODO define slots for all classes
# TODO tests transformations
class Transformer(Interpolator):

    @timer
    def __init__(self, m=2, n=2, lp_degree=2.0):
        Interpolator.__init__(self, m, n, lp_degree)
        self.__build_vandermonde_n2c()
        self.__build_transform_n2c()
        self.__build_vandermonde_l2n()
        self.__build_transform_l2n()
        self.__build_transform_l2c()

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
                    buildCanonicalVandermonde(x_in, gamma_placeholder).t())
        
        Parameters:
        X.. X positions
        gamma_placeholder.. gamma_placeholder vectors of our polynomial
        """
        noX, dimX = X.shape
        dimGamma, noCoefficients = gamma.shape
        assert dimX == dimGamma, "Input dimensions (%d,%d) of (X, gamma_placeholder) dont match." % (dimX, dimGamma)
        V = np.ones((noX, noCoefficients))
        # TODO: this probably only works for m == 2
        for j in range(noCoefficients):
            for k in range(dimX):
                V[:, j] *= (X[:, k] ** gamma[k, j])
            # V[:,j] = (X[:,0]**gamma_placeholder[0,j]) * (X[:,1]**gamma_placeholder[1,j])
        return V

    @timer
    def __build_vandermonde_n2c(self):
        self.V_n2c = np.ones((self.N, self.N))
        for i in np.arange(0, self.N):
            for j in np.arange(1, self.N):
                for d in np.arange(0, self.m):
                    self.V_n2c[i, j] = self.V_n2c[i, j] * self.tree.grid_points[d, i] ** self.exponents[d, j]

    @timer
    def __build_transform_n2c(self):
        self.Cmn_n2c = np.zeros((self.N, self.N))
        for j in np.arange(self.N):
            self.Cmn_n2c[:, j] = self.run(self.m, self.N, self.tree.tree, self.V_n2c[:, j].copy(),
                                          self.tree.grid_values.copy(),
                                          self.gamma_placeholder.copy(), 1, 1)
        self.newton2canonical = solve_triangular(self.Cmn_n2c, np.identity(self.N))
        self.canonical2newton = scipy.linalg.inv(self.newton2canonical)

    @timer
    def transform_n2c(self, coeffs_newton):
        coeffs_canonical = np.dot(self.newton2canonical, coeffs_newton)
        return coeffs_canonical

    @timer
    def __build_vandermonde_l2n(self):
        self.V_l2n = np.eye(self.V_n2c.shape[0])

    @timer
    def __build_transform_l2n(self):
        self.lagrange2newton = np.zeros((self.N, self.N))
        for j in np.arange(self.N):
            self.lagrange2newton[:, j] = self.run(self.m, self.N, self.tree.tree, self.V_l2n[:, j].copy(),
                                                  self.tree.grid_values.copy(),
                                                  self.gamma_placeholder.copy(), 1, 1)
        self.newton2lagrange = scipy.linalg.inv(self.lagrange2newton)

    # TODO should actually be defined in DDS ("interpolator" class)
    # TODO actually an alias for self.transform_l2n()
    def interpolate(self, fct_values):
        coeffs_newton_dds = self.run(self.m, self.N, self.tree.tree, fct_values, self.tree.grid_values,
                                     self.gamma_placeholder.copy(),
                                     1, 1)

        # can be computed by transformation as well
        # TODO question: what is more performant? prob. depends on if the conversion matrices have been built
        # function values on the interpolation grid are actually the lagrange coefficients
        # coeffs_newton_trans = self.transform_l2n(fct_values)
        # assert np.allclose(coeffs_newton_dds, coeffs_newton_trans)
        return coeffs_newton_dds

    @timer
    def transform_l2n(self, coeffs_lagrange):
        # return solve_triangular(self.Cmn_l2n,l)
        coeffs_newton = np.dot(self.lagrange2newton, coeffs_lagrange)
        return coeffs_newton

    @timer
    def transform_n2l(self, coeffs_newton):
        coeffs_lagrange = np.dot(self.newton2lagrange, coeffs_newton)
        return coeffs_lagrange

    @timer
    def __build_transform_l2c(self):
        self.lagrange2canonical = np.dot(self.newton2canonical, self.lagrange2newton)
        self.canonical2lagrange = np.dot(self.newton2lagrange, self.canonical2newton)

    @timer
    def transform_l2c(self, coeffs_lagrange):
        # return self.transform_n2c(self.transform_l2n(v))
        coeffs_canonical = np.dot(self.lagrange2canonical, coeffs_lagrange)
        return coeffs_canonical

    @timer
    def transform_c2l(self, coeffs_canonical):
        # return self.transform_n2c(self.transform_l2n(v))
        coeffs_lagrange = np.dot(self.canonical2lagrange, coeffs_canonical)
        return coeffs_lagrange

    # TODO static, no need to define this as a class function
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
                    buildCanonicalVandermonde(x_in, gamma_placeholder).t())

        Parameters:
        X.. X positions
        gamma_placeholder.. gamma_placeholder vectors of our polynomial
        """
        noX, dimX = X.shape
        dimGamma, noCoefficients = gamma.shape
        assert dimX == dimGamma, "Input dimensions (%d,%d) of (X, gamma_placeholder) dont match." % (dimX, dimGamma)
        V = np.ones((noX, noCoefficients))
        # TODO: this probably only works for m == 2
        for j in range(noCoefficients):
            for k in range(dimX):
                V[:, j] *= (X[:, k] ** gamma[k, j])
            # V[:,j] = (X[:,0]**gamma_placeholder[0,j]) * (X[:,1]**gamma_placeholder[1,j])
        return V


# TODO is this a test, routine...?
if __name__ == '__main__':
    import time

    times = {}
    np.random.seed(23232323)
    # m,n,lp_degree
    input_para = (2, 2, 2)

    startFull = time.time()

    start = time.time()
    test_tr = Transformer(*input_para)
    times['init Transformer'] = time.time() - start

    lagrange_coefs = np.zeros(test_tr.N)
    base_coefs = np.random.uniform(-10, 10, test_tr.N)
    start = time.time()
    for i in np.arange(test_tr.N):
        temp_lag = np.ones(test_tr.N)
        for j in np.arange(test_tr.N):
            for d in np.arange(test_tr.m):
                temp_lag[j] *= test_tr.tree.grid_points[d, i] ** test_tr.exponents[d, j]
            temp_lag[j] *= base_coefs[j]
        lagrange_coefs[i] = np.sum(temp_lag)
    times['build lagrange'] = time.time() - start

    start = time.time()
    newton = test_tr.transform_l2n(lagrange_coefs)
    times['Transformer to newton'] = time.time() - start

    start = time.time()
    # canon = transformer_from.transform_n2c(newton)
    canon = test_tr.transform_l2c(lagrange_coefs)
    times['Transformer to canon'] = time.time() - start

    TIMES['full'] = time.time() - startFull

    print("---- results ----")
    # print('base',base_coefs)
    # print('lagrange',lagrange_coefs)
    # print('newton',newton)
    # print("base again", canon)
    abs_err = np.abs(base_coefs - canon)
    print("max abs_err", abs_err.max())
    rel_err = np.abs(abs_err / (base_coefs + canon))
    print("max rel_err", rel_err.max())

    print("---- times ---- ")
    for key in times.keys():
        print(key, "\n\t%1.2es" % times[key])

    print("---- internal times ----")
    for key in TIMES.keys():
        print(key, "\n\t%1.2es" % TIMES[key])

    # print("full time:",times['build lagrange'] + sum(TIMES.values()))

    # base_coefs = np.random.uniform(-10,10,6)
    # g = lambda x: np.dot(base_coefs,np.array([1,x[0],x[0]**2,x[1],x[0]*x[1],x[1]**2]))
    # test_inter(g)
