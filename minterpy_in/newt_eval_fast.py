# -*- coding:utf-8 -*-

import numpy as np
from numba import f8, njit, i8, void

from minterpy.global_settings import FLOAT_DTYPE, INT_DTYPE

# NOTE must match the numpy data types specified in global_settings.py
f = f8
f_list_1D = f[:]
f_list_2D = f[:, :]
i = i8
i_list_1D = i[:]
i_list_2D = i[:, :]


# NOTE: the most "fine grained" functions must be defined first
# in order for Numba to properly infer the function types

# O(N)
@njit(f(f_list_1D, f_list_1D), cache=True)
def single_eval(coefficients, monomial_vals):
    # single eval with a single point and a single list of coefficients
    assert len(coefficients) == len(monomial_vals)
    # the value of a polynomial in Newton form
    # is the sum over all coefficients multiplied with the value of the corresponding Newton polynomial
    return np.sum(coefficients * monomial_vals)


@njit(void(f_list_1D, i_list_2D, f_list_2D, i_list_1D, f_list_2D, f_list_1D), cache=True)
def eval_newton_polynomials(x, exponents, grid_values, max_exponents, prod_placeholder, monomial_vals_placeholder):
    """ precomputes the value of all N Newton polynomials at a fixed point x

    core of the fast polynomial evaluation algorithm

    NOTE: coefficient agnostic
        precompute all the (chained) products required during evaluation. O(mn)
    NOTE: the maximal exponent might be different in every dimension,
        in this case the matrix becomes sparse (towards the end)
    NOTE: avoid index shifting during evaluation (has larger complexity than pre-computation!)
        by just adding one empty row in front. ATTENTION: these values must not be accessed!
        -> the exponents of each monomial ("alpha") then match the indices of the required products

    Parameters
    ----------
    prod_placeholder: a numpy array for storing the (chained) products
    monomial_vals_placeholder: a numpy array of length N for storing the values of all Newton polynomials

    Returns
    -------
    None
    """

    m = len(x)
    # precompute all occurring products O(mn)
    for i in range(m):  # O(m)
        max_exponent = max_exponents[i]
        x_i = x[i]
        prod = 1.0
        for j in range(max_exponent):  # O(n)
            # TODO there are n+1 1D grid values, the last one will never be used!?
            p_ij = grid_values[i, j]
            prod *= (x_i - p_ij)
            # NOTE: shift index by one
            exponent = j + 1  # NOTE: otherwise the result type is float
            prod_placeholder[i, exponent] = prod

    # evaluate all Newton polynomials. O(Nm)
    N = exponents.shape[1]
    for j in range(N):  # O(N)
        # the exponents of each monomial ("alpha")
        # are the indices of the products which need to be multiplied
        newt_mon_val = 1.0  # one required as multiplicative identity
        for i in range(m):  # O(m)
            exp = exponents[i, j]
            # NOTE: an exponent of 0 should not cause a multiplication
            # (inefficient, numerical instabilities)
            if exp > 0:
                newt_mon_val *= prod_placeholder[i, exp]
        monomial_vals_placeholder[j] = newt_mon_val
    # NOTE: results have been stored in the numpy arrays. no need to return anything.


@njit(void(f_list_2D, f_list_2D, i_list_2D, f_list_2D, i_list_1D, f_list_2D, f_list_1D, f_list_2D), cache=True)
def evaluate_multiple(x, coefficients, exponents, grid_values, max_exponents, prod_placeholder,
                      monomial_vals_placeholder, results_placeholder):
    nr_points = x.shape[1]
    nr_polynomials = coefficients.shape[1]
    for point_nr in range(nr_points):
        x_single = x[:, point_nr]
        # NOTE: with a fixed single point x to evaluate the polynomial on,
        # the values of the Newton polynomials become fixed (coefficient agnostic)
        # -> precompute all intermediary results (=compute the value of all Newton polynomials)
        eval_newton_polynomials(x_single, exponents, grid_values, max_exponents, prod_placeholder,
                                monomial_vals_placeholder)
        for poly_nr in range(nr_polynomials):
            coeffs_single = coefficients[:, poly_nr]
            results_placeholder[point_nr, poly_nr] = single_eval(coeffs_single, monomial_vals_placeholder)


# TODO idea for improvement: make use of the sparsity of the exponent matrix and avoid iterating over the zero entries!
# TODO rectify usage of the term "monomial", there are no Lagrange monomials,
#  the literature calls them Lagrange Polynomials. and p(x) a polynomial in Lagrange basis. distinguish.
def newt_eval(x, coefficients, exponents, grid_values, verify_input: bool = False):
    """ iterative implementation of polynomial evaluation in Newton form

    version able to handle both:
     - list of input points x (2D input)
     - list of input coefficients (2D input)

     NOTE: assuming equal input array shapes as the reference implementation


     n = polynomial degree
     N = amount of coefficients
     k = amount of points
     p = amount of polynomials


    faster than the recursive implementation of tree.eval_lp(...)
    for a single point and a single polynomial (1 set of coeffs):
    time complexity: O(mn+mN) = O(m(n+N)) = O(mN)
        pre-computations: O(mn)
        evaluation: O(mN)
        NOTE: N >= n and N >> n depending on m and the l_p-degree

    space complexity: O(mn+N) = O(N)
        storing the products: O(mn)
        evaluating the lagrange polynomials: O(N)
        NOTE: N >= nm

    advantage:
        - just operating on numpy arrays, can be just-in-time (jit) compiled
        - can evaluate multiple polynomials without recomputing all intermediary results


    Parameters
    ----------
    x: (m x k) the k points to evaluate on with dimensionality m.
    coefficients: (N x p) the coefficients of the Newton polynomials.
        NOTE: format fixed such that 'lagrange2newton' conversion matrices can be passed
        as the Newton coefficients of all Lagrange monomials of a polynomial without prior transponation
    exponents: (m x N) a multi index "alpha" for every Newton polynomial
        corresponding to the exponents of this "monomial"
    grid_values: (m x n+1) Leja ordered Chebychev values for every dimension.
        the values determine the locations of the hyperplanes of the interpolation grid.
        the ordering of the values determine the spacial distribution of interpolation nodes.
        (relevant for the approximation properties and the numerical stability).
    verify_input: weather the datatypes of the input should be checked. turned off by default for speed.


    Returns
    -------
    (k x p) the value of each input polynomial in Newton form at each points.
    squeezed into the expeced shape (1D if possible)
    NOTE: format fixed such that the regression can use the result as transformation matrix without transponation
    """
    m, N = exponents.shape
    # NOTE: silently reshaping the input is dangerous,
    # because the value order could be changed without being noticed by the user -> unexpected results!
    query_point_shape = x.shape
    if len(query_point_shape) == 1:
        if m == 1:  # -> every entry is a query point
            nr_points = query_point_shape[0]
        else:  # m > 1, interpret input as a single point -> dimensions must match!
            assert len(x) == m, f'points x given as vector of shape {query_point_shape} (1D). ' \
                                f'detected dimensionality of the exponents however is {m}'
            nr_points = 1
        x = x.reshape(m, nr_points)  # reshape to 2D
    else:
        m_points, nr_points = x.shape
        assert m == m_points, \
            f'dimensionality of the input points {m_points} does not match the polynomial dimensionality {m}'
    coeff_shape = coefficients.shape
    if len(coeff_shape) == 1:  # 1D: a single query polynomial
        nr_polynomials = 1
        coefficients = coefficients.reshape(N, nr_polynomials)  # reshape to 2D
    else:
        N_coeffs, nr_polynomials = coefficients.shape
        assert N == N_coeffs, \
            f'coefficient amount {N_coeffs} does not match the amount of monomials {N}'
    # TODO possible with just (1D) Leja values, ATTENTION: sign depending on dimension!
    m_grid, nr_grid_values = grid_values.shape
    max_exponents = np.max(exponents, axis=1)

    if verify_input:
        assert x.dtype == FLOAT_DTYPE, f'query point x: expected dtype {FLOAT_DTYPE} got {x.dtype}'
        assert coefficients.dtype == FLOAT_DTYPE, f'coefficients: expected dtype {FLOAT_DTYPE} got {x.dtype}'
        assert grid_values.dtype == FLOAT_DTYPE, f'grid values: expected dtype {FLOAT_DTYPE} got {x.dtype}'
        assert exponents.dtype == INT_DTYPE, f'exponents: expected dtype {INT_DTYPE} got {x.dtype}'
        # DEBUG only: should be correct when Tree class implementation is correct
        # assert m == m_grid, f'the dimensionality {m} of the input exponents, does not match the grid point exponents'
        # assert nr_grid_values == n + 1, 'the grid values do not match the exponents. there must be a n+1 grid values.'

    # initialise arrays required for computing and storing the intermediary results:
    # will be reused in the different runs -> memory efficiency
    prod_placeholder = np.empty((m, nr_grid_values), dtype=FLOAT_DTYPE)
    monomial_vals_placeholder = np.empty(N, dtype=FLOAT_DTYPE)
    results_placeholder = np.empty((nr_points, nr_polynomials), dtype=FLOAT_DTYPE)
    evaluate_multiple(x, coefficients, exponents, grid_values, max_exponents,
                      prod_placeholder, monomial_vals_placeholder, results_placeholder)
    return results_placeholder.squeeze()  # convert into the expected shape
