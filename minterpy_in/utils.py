# -*- coding:utf-8 -*-
from typing import Iterator, List

import numpy as np


def report_error(errors, description=None):
    if description is not None:
        print('\n\n')
        print(description)

    print(f'mean: {np.mean(errors):.2e}')
    print(f'median: {np.median(errors):.2e}')
    print(f'variance: {np.var(errors):.2e}')
    print(f'l2-norm: {np.linalg.norm(errors):.2e}')
    # f"l_infty error (max): {np.linalg.norm(errors, ord=np.inf)}\n")
    errors = np.abs(errors)
    print(f'abs mean: {np.mean(errors):.2e}')
    print(f'abs median: {np.median(errors):.2e}')
    print(f'abs variance: {np.var(errors):.2e}')


def chebpoints_2nd_order(n):  # 2nd order
    # TODO dtype not useful when all other computations are being performed with float64.
    #  causes type conversions
    return np.cos(np.arange(n, dtype=np.float128) * np.pi / (n - 1))


def rnd_points(*shape):
    return 2 * (np.random.rand(*shape) - 0.5)  # [-1;1]


def leja_ordered_values(n):
    points1 = chebpoints_2nd_order(n + 1)[::-1]
    points2 = points1  # TODO
    ord = np.arange(1, n + 1)

    lj = np.zeros([1, n + 1])
    lj[0] = 0
    m = 0

    for k in range(0, n):
        jj = 0
        for i in range(0, n - k):
            P = 1
            for j in range(k + 1):
                idx_pts = int(lj[0, j])
                P = P * (points1[idx_pts] - points1[ord[i]])
            P = np.abs(P)
            if (P >= m):
                jj = i
                m = P
        m = 0
        lj[0, k + 1] = ord[jj]
        ord = np.delete(ord, jj)

    leja_points = np.zeros([1, n + 1])
    for i in range(n + 1):
        leja_points[0, i] = points2[int(lj[0, i])]
    return leja_points


# build multi index "gamma" depending on lp_degree
# NOTE: multi indices are being called "alpha" in the newest interpolation paper
def gamma_lp(m, n, gamma, gamma2, lp_degree):
    # TODO better parameter and variable names
    # TODO cleanup. input parameters...
    # TODO efficient implementation, pre allocate memory, jit compilation...
    gamma0 = gamma.copy()
    gamma0[m - 1] = gamma0[m - 1] + 1

    out = []
    norm = np.linalg.norm(gamma0.reshape(-1), lp_degree)
    if norm < n and m > 1:
        o1 = gamma_lp(m - 1, n, gamma.copy(), gamma.copy(), lp_degree)
        o2 = gamma_lp(m, n, gamma0.copy(), gamma0.copy(), lp_degree)
        out = np.concatenate([o1, o2], axis=-1)
    elif norm < n and m == 1:
        out = np.concatenate([gamma2, gamma_lp(m, n, gamma0.copy(), gamma0.copy(), lp_degree)], axis=-1)
    elif norm == n and m > 1:
        out = np.concatenate([gamma_lp(m - 1, n, gamma.copy(), gamma.copy(), lp_degree), gamma0], axis=-1)
    elif norm == n and m == 1:
        out = np.concatenate([gamma2, gamma0], axis=-1)
    elif norm > n:
        norm_ = np.linalg.norm(gamma.reshape(-1), lp_degree)
        if norm_ < n and m > 1:
            for j in range(1, m):
                gamma0 = gamma.copy()
                gamma0[j - 1] = gamma0[j - 1] + 1  # gamm0 -> 1121 broken
                if np.linalg.norm(gamma0.reshape(-1), lp_degree) <= n:
                    gamma2 = np.concatenate([gamma2, gamma_lp(j, n, gamma0.copy(), gamma0.copy(), lp_degree)], axis=-1)
            out = gamma2
        elif m == 1:
            out = gamma2
        elif norm_ <= n:
            out = gamma

    return out


def find_matching_indices(subset_indices, indices):
    matching_indices = []
    for i1 in subset_indices.T:
        idx = -1
        for n, i2 in enumerate(indices.T):
            if np.array_equal(i1, i2):
                idx = n
                break

        if idx == -1:
            raise ValueError(f'{i1} multi index was not contained!')
        matching_indices.append(idx)

    return matching_indices


def get_eval_fct(tree, coeffs_newton):
    return lambda x: tree.eval(x, coeffs_newton)


# TODO remove. simplifications possible. iterative evaluation is able to handle multiple points
def apply_vectorized(eval_fct, eval_points):
    return np.apply_along_axis(eval_fct, 0, eval_points)


def eval_fct_canonical(x, coefficients, exponents):
    coeffs_copy = coefficients.copy()
    nr_coeffs = len(coefficients)
    nr_dims, nr_monomials = exponents.shape
    assert nr_monomials == nr_coeffs
    assert len(x) == nr_dims
    for i in range(nr_coeffs):
        coeffs_copy[i] = coeffs_copy[i] * np.prod(np.power(x, exponents[:, i]))

    return np.sum(coeffs_copy)


def get_eval_fct_canonical(coefficients, exponents):
    # return lambda x: np.sum(coefficients.T * np.prod(np.power(x, exponents), axis=1), axis=1)[0]
    return lambda x: eval_fct_canonical(x, coefficients, exponents)


def eval_on_lagrange(sample_pts: np.ndarray, coeffs_lagrange: np.ndarray, transformer):
    coeffs_newton = transformer.transform_l2c(coeffs_lagrange)
    return transformer.tree.eval(sample_pts, coeffs_newton)


# TODO later: use adapted regression evaluation. precompute "evaluation matrices"
def rescale(coeffs_lagrange: np.ndarray, transformer_from, transformer_to):
    new_grid = transformer_to.grid_points
    coeffs_lagrange_new = eval_on_lagrange(new_grid, coeffs_lagrange, transformer_from)
    return coeffs_lagrange_new


def assert_shape(array_iter: Iterator[np.ndarray], shape):
    for a in array_iter:
        assert a.shape == shape, f'encountered shape {a.shape} different from expected shape {shape}'


def assert_all_equal(arrays: List[np.ndarray]):
    a0 = arrays.pop()
    for a in arrays:
        np.testing.assert_allclose(a0, a, atol=1e-8)
        # assert np.allclose(a0, a)


# TODO is this a test, routine...?
if __name__ == '__main__':
    import scipy  # required where?

    print("scipy version", scipy.__version__)
    import matplotlib.pylab as plt  # TODO add to project requirements?

    import h5py  # TODO add to project requirements?

    with h5py.File("chebpts.mat", 'r') as chebpts:
        cp_10 = np.asarray(chebpts['cp_10'])[0]
        cp_50 = np.asarray(chebpts['cp_50'])[0]
        cp_100 = np.asarray(chebpts['cp_100'])[0]
        cp_500 = np.asarray(chebpts['cp_500'])[0]
        cp_1000 = np.asarray(chebpts['cp_1000'])[0]
        chebfun_points = [cp_10, cp_50, cp_100, cp_500, cp_1000]

    n_arr = np.array([10, 50, 100, 500, 1000])
    for i, n in enumerate(n_arr):
        a = chebfun_points[i]
        b = chebpoints_2nd_order(n)[::-1]  # roots_chebyt(n)[0]
        # b=roots_chebyu(n)[0]
        abs_err = np.abs(a - b)
        rel_err = abs_err / (np.abs(a + b))
        print('mean abs err', abs_err.mean())
        print('mean rel err', rel_err.mean())
        plt.plot(n, abs_err.mean(), 'or')
        plt.plot(n, rel_err.mean(), 'Xk')
        plt.plot(n, abs_err.max(), '>g')

    plt.yscale('log')
    plt.show()

import numpy as np
