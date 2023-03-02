# -*- coding:utf-8 -*-
from itertools import product
from math import floor
from random import randint

import numpy as np

from minterpy.regression import MultivariatePolynomialRegression
from minterpy.tree import MultiIndicesTree
from minterpy.unit_test import RUNGE_FCT_VECTORIZED
# TODO integrate function for "down-" and "upsampling" polynomial base (=grid!) into Interpolator class
#  .change_degree(n_diff)
from minterpy.utils import rnd_points, report_error

np.random.seed(42)

MIN_DEGREE = 1
MAX_DEGREE = 3

# ONLY_UNEVEN_DEGREES = True  # test only uneven interpolation total_degrees (result in more symmetrical grid)
ONLY_UNEVEN_DEGREES = False
if ONLY_UNEVEN_DEGREES:
    assert MIN_DEGREE % 2 == 1
    assert MAX_DEGREE % 2 == 1
    DEGREE_STEP = 2
else:
    DEGREE_STEP = 1

MIN_DIMENSION = 1
MAX_DIMENSION = 3

SAMPLING_RATIOS = [0.5, 1.5]

NUMERICAL_TOLERANCE = 1e-10


def eval_fit(test_points, regressor, fct_ground_truth, test_point_descr='interpolation nodes'):
    def eval_error(points, test_point_descr):
        m, N_pts = points.shape
        print(f'\nerror on {N_pts} {test_point_descr}:')
        F_true = fct_ground_truth(points)
        F_estim = regressor.evaluate_on(points)
        err = F_estim - F_true
        report_error(err)

    m, N_pts = test_points.shape
    assert m == regressor.m
    eval_error(test_points, test_point_descr)
    random_pts = rnd_points(m, N_pts)  # same amount as samples to avoid bias
    eval_error(random_pts, 'random points')
    print('\n')


def equality2interpolation_test(m=3, n=4):
    print('\ntesting equality to interpolation')
    regressor = MultivariatePolynomialRegression(m, n)
    transformer = regressor.transformer
    # equal to lagrange coefficients
    N = regressor.N_fit
    fct_values = rnd_points(N)
    coeffs_newton_interpol = transformer.interpolate(fct_values)
    sample_points = transformer.grid_points

    def test(regressor):
        np.testing.assert_allclose(fct_values, regressor.fct_values)
        vals_on_samples = regressor.fct_values_regr
        error_regr = regressor.fct_val_errors
        coeffs_lagr_regr = regressor.coeffs_lagrange
        np.testing.assert_allclose(vals_on_samples, coeffs_lagr_regr), \
        'since the sample points of the regression are equal to the interpolation grid, ' \
        'the values on the samples and the Lagrange coefficients of the polynomial should be equal!'

        np.testing.assert_allclose(error_regr, 0.0), \
        'for input polynomials the regression error should be machine prevision'

        # a (close to) zero error, should mean that the computed coefficients are equal to the input values
        np.testing.assert_allclose(fct_values, coeffs_lagr_regr), \
        'the regression of a polynomial with degree n' \
        'with the interpolation grid as input (samples)' \
        'should yield the same result as the interpolation'

        np.testing.assert_allclose(regressor.R, np.eye(N), rtol=NUMERICAL_TOLERANCE, atol=NUMERICAL_TOLERANCE), \
        'the transformation matrix should be the identity'

        coeffs_newt_reg = transformer.transform_l2n(coeffs_lagr_regr)
        np.testing.assert_allclose(coeffs_newt_reg, coeffs_newton_interpol)

    # TODO test if calling regression without sample points and no stored transformation raises Assertion error

    # test computing the transformation and the fit (regression) separately
    regressor.cache_transform(sample_points)
    # test if regression works without passing sample points
    regressor.regression(fct_values)
    test(regressor)

    # should be the same as computing both at the same time
    regressor.regression(fct_values, sample_points, use_cached_transform=False)
    test(regressor)

    print('OK.')


# TODO define test condition
# TODO not only regress on interpolation grid points (of different degree)
def interpol_grid_regr_test(sampling_ratio, m, n):
    # uses data sampled on a regular Chebychev interpolation grid (not scattered)
    # sampling_ratio > 1 -> "oversampling"
    # sampling_ratio < 1 -> "undersampling"

    n_fit = n  # degree
    interpolation_grid = MultiIndicesTree(m=m, n=n_fit).grid_points
    N_fit = interpolation_grid.shape[1]
    nr_points2sample = max(1, floor(N_fit * sampling_ratio))

    # find the smallest degree for which there are more grid points
    N_data = 1
    n_data = 1
    # TODO scaling factor?!, allow some extrapolation
    samples = MultiIndicesTree(m=m, n=n_data).grid_points

    while N_data < nr_points2sample:
        n_data += 1
        samples = MultiIndicesTree(m=m, n=n_data).grid_points
        N_data = samples.shape[1]

    selection_idxs = np.random.choice(N_data, size=nr_points2sample, replace=False)
    samples = samples[:, selection_idxs]
    N_data = nr_points2sample

    if N_fit > N_data:
        # (= more interpolation nodes / higher degree (of freedom) than interpolation data)
        print('\n\nTESTING OVERFITTING:')
    elif N_fit < N_data:
        print('\n\nTESTING UNDERFITTING:')
    print(f'sampling ratio {sampling_ratio}')  # (= more interpolation nodes/ higher degree than interpolation data)
    print(f'{N_fit} polynomial coefficients (polynomial degree {n_fit})')
    print(f'{N_data} data samples (grid points from polynomial of degree {n_data})')

    # if N_fit == N_data:
    #     raise ValueError('trying to fit the same amount of data')

    # ground truth
    # TODO hyperparam benchmark
    fct_ground_truth = RUNGE_FCT_VECTORIZED
    F_gt_data = fct_ground_truth(interpolation_grid)
    F_gt_sampled = fct_ground_truth(samples)

    # NOTE: different degree! defining the degree of the output polynomial
    regressor = MultivariatePolynomialRegression(m=m, n=n_fit, verbose=True)

    fct_vals_regr, err, coeffs_lagrange = regressor.regression(F_gt_sampled, samples)
    eval_fit(interpolation_grid, regressor, fct_ground_truth)

    # NOTE: samples remain equal. no need to compute transformation again (samples=None)
    fct_vals_regr, err, coeffs_lagrange = regressor.simple_regression(F_gt_sampled)
    eval_fit(interpolation_grid, regressor, fct_ground_truth)

    # print(f'S: {regressor.S.shape}')
    # except np.linalg.LinAlgError:
    #     warnings.warn('the regression was numerically unstable')

    print('tests passed.')


def test_weighted_regr(m, n):
    print('\ntesting equality to interpolation')
    regressor = MultivariatePolynomialRegression(m, n)
    transformer = regressor.transformer
    # equal to lagrange coefficients
    N = regressor.N_fit
    fct_values = rnd_points(N)
    weights = np.abs(rnd_points(N)) # weights must be positive
    sample_points = transformer.grid_points
    regressor.weighted_regression(fct_values, weights, sample_points, verify_input=True)
    print('tests passed.')


def test_lagrange_monomial_eval(m, n):
    print('\ntesting the evaluation of Lagrange monomial gradient...')
    # evaluating the Lagrange monomials on the corresponding interpolation grid should yield the identity matrix
    regressor = MultivariatePolynomialRegression(m, n)
    interpol_nodes = regressor.transformer.tree.grid_points
    monomial_vals = regressor.eval_lagrange_monomials_on(interpol_nodes)
    np.testing.assert_allclose(monomial_vals, np.eye(regressor.N_fit), rtol=NUMERICAL_TOLERANCE,
                               atol=NUMERICAL_TOLERANCE)

    coeffs_lagrange = rnd_points(regressor.N_fit)
    regressor.simple_regression(fct_values=coeffs_lagrange, sample_points=interpol_nodes)

    # evaluating the polynomial in Newton and in Lagrange form should yield the same result:
    nr_rnd_points = regressor.N_fit - 1  # less or equal points otherwise there will be an approximation error
    scattered_points = rnd_points(m, nr_rnd_points)
    vals_newton = regressor.evaluate_on(scattered_points)
    monomial_vals = regressor.eval_lagrange_monomials_on(scattered_points)
    vals_lagrange = monomial_vals @ coeffs_lagrange
    np.testing.assert_allclose(vals_lagrange, vals_newton)

    fct_values = rnd_points(nr_rnd_points)
    regressor.simple_regression(fct_values=fct_values, sample_points=scattered_points)
    vals_newton = regressor.evaluate_on(scattered_points)
    np.testing.assert_allclose(fct_values, vals_newton)  # regression works
    monomial_vals = regressor.eval_lagrange_monomials_on(scattered_points)
    coeffs_lagrange = regressor.coeffs_lagrange
    vals_lagrange = monomial_vals @ coeffs_lagrange
    np.testing.assert_allclose(vals_lagrange, vals_newton)


def test_lagrange_monomial_gradient_eval(m, n):
    print('\ntesting the evaluation of Lagrange monomial gradient...')
    regressor = MultivariatePolynomialRegression(m, n)
    point = rnd_points(m)
    # should automatically initialise gradient!
    grad_eval = regressor.eval_lagrange_monomial_gradient_on(point)
    m_grad, N, _ = regressor.grad_op_l2n.shape
    assert regressor.N_fit == N
    assert m == m_grad
    grad_eval_ref = np.empty((m, N))
    for i in range(m):
        gradient_coeffs = regressor.grad_op_l2n[i]
        grad_eval_ref[i] = regressor.transformer.tree.eval(point, gradient_coeffs)
    np.testing.assert_allclose(grad_eval_ref, grad_eval)


def test_adding_point(m, n):
    print('\ntesting to add a single data point to a transformation (regression)...')
    regressor = MultivariatePolynomialRegression(m, n)
    points = regressor.transformer.tree.grid_points
    N = points.shape[1]
    assert N == regressor.N_fit
    # NOTE: add last point and test if transformation matrix is equal.
    # required since new entries will be appended
    point = points[:, -1]  # last point
    regressor.cache_transform(points[:, :-1])
    # should work without any instabilities
    regressor.add_point(point)
    np.testing.assert_allclose(regressor.R, np.eye(N), rtol=NUMERICAL_TOLERANCE, atol=NUMERICAL_TOLERANCE)
    np.testing.assert_allclose(regressor.S, np.eye(N), rtol=NUMERICAL_TOLERANCE, atol=NUMERICAL_TOLERANCE)
    np.testing.assert_allclose(regressor.sample_points, points)
    assert regressor.transformation_stored


def test_proposing_point(m, n):
    print('\ntesting to generate point suggestions...')
    # the method should suggest "missing interpolation node" as data samples
    regressor = MultivariatePolynomialRegression(m, n)
    points = regressor.transformer.tree.grid_points
    N = points.shape[1]
    assert N == regressor.N_fit
    pt_idx = randint(0, N - 1)
    point = points[:, pt_idx]
    points = np.delete(points, pt_idx, axis=1)
    regressor.cache_transform(points)
    suggested_point = regressor.propose_n_add_point()
    np.testing.assert_allclose(suggested_point, point)
    # afterwards the transformation should be "complete" = shuffled identity
    np.testing.assert_allclose(np.max(regressor.R, axis=0), 1.0)
    np.testing.assert_allclose(np.max(regressor.R, axis=1), 1.0)


def test_different_settings():
    for m, n, in product(range(MIN_DIMENSION, MAX_DIMENSION + 1),
                         range(MIN_DEGREE, MAX_DEGREE + 1, DEGREE_STEP),
                         ):
        print(f'\n\ndim {m}, degree {n}')
        equality2interpolation_test(m, n)
        test_lagrange_monomial_eval(m, n)
        test_lagrange_monomial_gradient_eval(m, n)
        test_adding_point(m, n)
        test_proposing_point(m, n)
        test_weighted_regr(m, n)

        for sampling_ratio in SAMPLING_RATIOS:
            interpol_grid_regr_test(sampling_ratio, m, n)

    print('... OK.\n\n')


# TODO general test case

# deprecated:
# print('\n\nsimple regression:')
# F_hat, err, H_ref = regressor.simple_regression(F_gt_subsampled, chebychev_grid_subsampled)
# eval_error(H_ref, chebychev_grid_regular)

# TODO avg
# degrees = [15]
# for d in degrees:
#     start_time = time.time()
#     regressor = MPR(m=2,n=d,lp_degree=2)


# TODO unit tests
# TODO test the different regression functions (simple...)!
# TODO compare regression results of diff. regr. fcts.
# TODO test on scattered data, define test condition, handle case of singular transformation matrices
# TODO test adding a single sample data point to the tranformation
# TODO test gradient
if __name__ == '__main__':
    test_different_settings()
