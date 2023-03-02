# -*- coding:utf-8 -*-
import warnings

import numpy as np

from minterpy.tree import MultiIndicesTree


def dds_1d(grid_values, fct_values):
    # https://en.wikipedia.org/wiki/Divided_differences
    # https://stackoverflow.com/questions/14823891/newton-s-interpolating-polynomial-python
    n = len(grid_values)
    grid_values = np.copy(grid_values)
    c = np.copy(fct_values)  # newton coefficients
    for k in range(1, n):
        c[k:] = (c[k:] - c[k - 1]) / (grid_values[k:] - grid_values[k - 1])
    return c


# TODO define slots for all classes
# TODO pre- or just in time compile the core algorithms, implement in C...
# TODO required as own class? used standalone? combine with Interpolator?
# TODO only run fct. refactor. use class attributes
class DDS:

    # TODO
    def __init__(self, *args, **kwargs):
        pass

    # TODO explain parameters. especially I, J
    # TODO API wrapped in transformation.interpolate()
    # TODO points, cheby (1D) or leja values (2D)?!
    def run(self, m, N, tree, fct_values, grid_values, gamma, I, J):
        if m == 1:  # 1D DDS
            # NOTE: the function values get split up. the grid values however do not. take the first N Leja grid values
            return dds_1d(grid_values[0, :N], fct_values)

        # TODO the "gamma" parameter is just a placeholder for exponents
        #  (called "gamma_placeholder" in the Interpolator class)
        assert np.issubdtype(gamma.dtype, np.int_), \
            'the parameter "gamma" (exponent placeholder) should be given in integer dtype (used for indexing)'
        # ATTENTION: independent copy needed!
        # NOTE: with this there is no need to copy any additional imput parameters
        fct_values = fct_values.copy()
        gamma = gamma.copy()

        # nD DDS
        N0 = tree[(I, J)].split[0]  # left subtree
        N1 = tree[(I, J)].split[1]  # right subtree
        # split up the function values:
        F0 = fct_values[:N0]
        F1 = fct_values[N0:N]

        # Project F0 onto F1 and compute(F1 - F0) / QH
        s0 = 1
        s1 = 0
        pn = 0

        # Traverse binary tree
        gamma_constant = gamma[m - 1]
        # TODO remove int type conversions everywhere
        if N1 > 1:

            PF0 = F0
            # PF0 = np.arange(len(F0))

            pn = tree[(I, J)].pro_number
            for i in range(pn):
                PF0_index2 = np.ones(len(PF0), dtype=bool)
                Pro = tree[(I, J)].project[(1, i + 1)]
                k0 = Pro[0]
                s1 = s1 + Pro[2]

                if k0 > 0:
                    l = 0  # TODO
                    jj = np.arange(int(k0), dtype=np.int)
                    PF0_index2[Pro[3 + jj].astype('int') - 1] = False
                    # for j in range(int(k0)):
                    #    PF0 = np.delete(PF0, int(Pro[3 + j]) - l - 1)
                    #    l = l + 1
                QH = grid_values[m - 1, gamma_constant + i + 1] - grid_values[m - 1, gamma_constant]
                PF0 = PF0[PF0_index2]

                F1[int(s0) - 1:int(s1)] = (F1[int(s0) - 1:int(s1)] - PF0[0:int(Pro[2])]) / QH
                s0 = s0 + Pro[2]

        # Substract the constant part if existing
        if s1 < N1 or N1 == 1:
            QH = grid_values[m - 1, gamma_constant + pn + 1] - grid_values[m - 1, gamma_constant]
            F1[N1 - 1] = (F1[N1 - 1] - F0[0]) / QH

        # Recursion
        gamma[m - 1] = gamma_constant + 1

        if m > 1 and pn >= 1:
            tree_child1 = tree[(I, J)].child[0]
            tree_child2 = tree[(I, J)].child[1]
            o1 = self.run(m - 1, N0, tree, F0, grid_values, gamma, I + 1, tree_child1)
            o2 = self.run(m, N1, tree, F1, grid_values, gamma, I + 1, tree_child2)
            out = np.concatenate([o1, o2], axis=-1)
        elif m > 1:
            tree_child1 = tree[(I, J)].child[0]
            o1 = self.run(m - 1, N0, tree, F0, grid_values, gamma, I + 1, tree_child1)
            out = np.concatenate([o1, F1], axis=-1)
        elif m == 1 and pn >= 1:
            tree_child2 = tree[(I, J)].child[1]
            o2 = self.run(m, N1, tree, F1, grid_values, gamma, I + 1, tree_child2)
            out = np.concatenate([F0, o2], axis=-1)
        else:
            out = np.concatenate([F0, F1], axis=-1)

        return out


# TODO rename: newton solver?!, pip solver?!
# TODO test

# TODO rather look interpolation result: -> class Polynomial
# TODO store coefficients directly in this class
# TODO function eval. integrate multivar_horner
# TODO add functionality for automatically adding two polynomials
# TODO merge with "Transformer"

class Interpolator(DDS):
    def __init__(self, m: int = 2, n: int = 2, lp_degree: float = 2.0):
        DDS.__init__(self)  # TODO unused
        if n < 1:
            raise ValueError('the interpolation degree has to be at least 1.')
        if m < 1:
            raise ValueError('the dimensionality has to be at least 1.')
        if m > 10:
            warnings.warn(f'high dimensionality {m} detected. expect long run times.', category=UserWarning)
        if n > 10:
            warnings.warn(f'high degree {n} detected. expect long run times.', category=UserWarning)
        self.m = m
        self.n = n
        self.lp_degree = lp_degree
        # TODO no need to store and copy empty placeholder
        self.gamma_placeholder = np.zeros(self.m, dtype=np.int_)
        self.tree = MultiIndicesTree(m=self.m, n=self.n, lp_degree=self.lp_degree)

        # TODO redundancy
        self.N = self.tree.N
        self.exponents = self.tree.gamma
        self.grid_points = self.tree.grid_points

    def __call__(self, func):
        F = np.zeros([self.N])
        for i in range(self.N):
            F[i] = func(self.tree.grid_points[:, i])
        self.D = self.run(self.m, self.N, self.tree.tree, F, self.tree.grid_values,
                          self.gamma_placeholder, 1, 1)
        return lambda x: self.tree.eval(x, self.D)
