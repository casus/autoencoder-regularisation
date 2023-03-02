# -*- coding:utf-8 -*-
import timeit
from itertools import product

import numpy as np

from minterpy.newt_eval_fast import newt_eval
from minterpy.transformation import Transformer
from minterpy.tree import MultiIndicesTree
# TODO integrate function for "down-" and "upsampling" polynomial base (=grid!) into Interpolator class
#  .change_degree(n_diff)
from minterpy.utils import rnd_points, report_error

np.random.seed(42)
ATOL = 1e-7
RTOL = 1e-7

MIN_DEGREE = 1
MAX_DEGREE = 4

# TODO make tests work for m==1
# TODO only new eval fct?!
MIN_DIMENSION = 2
MAX_DIMENSION = 4

NR_SPEED_SAMPLES = int(1e2)
# DO_SPEED_TESTS = True
DO_SPEED_TESTS = False

NR_NUMERICAL_SAMPLES = int(1e2)
# DO_NUMERICAL_TESTS = True
DO_NUMERICAL_TESTS = False


def time_speedup(fct1, fct2, nr_runs=NR_SPEED_SAMPLES):
    t_recursive = timeit.timeit("a()", globals={'a': fct1}, number=nr_runs)
    print(f'{t_recursive:.2e}s recursive tree evaluation')
    t_iterative = timeit.timeit("b()", globals={'b': fct2}, number=nr_runs)
    print(f'{t_iterative:.2e}s iterative evaluation')
    t_diff = t_recursive - t_iterative
    if t_diff > 0.0:  # iter faster than recursive
        speed = t_diff / t_iterative
        print(f'{round(speed, 2)} x faster ({t_diff:.1e}s speedup)')
    else:
        t_diff = abs(t_diff)
        speed = t_diff / t_recursive
        print(f'{round(speed, 2)} x slower ({t_diff:.1e}s slowdown)')


def test_num_stability(val_diff_fct, nr_runs=NR_SPEED_SAMPLES):
    errors = np.empty(nr_runs)
    for i in range(nr_runs):
        errors[i] = val_diff_fct()
    report_error(errors)


def test_equality2recursive_eval(m, n):
    print('testing equality of iterative eval implementation to the recursive formulation')
    # NOTE: recursive algorithm is only working for a single polynomial (one set of N coefficients)
    tree = MultiIndicesTree(m, n)
    N = tree.N
    exponents = tree.gamma
    grid_values = tree.grid_values
    coeffs_newton = rnd_points(shape=N)
    x = rnd_points(shape=m)

    def recursive_eval(x=x, coeffs_newton=coeffs_newton):
        return tree.eval(x, coeffs_newton)

    def iterative_eval(x=x, coeffs_newton=coeffs_newton):
        return newt_eval(x, coeffs_newton, exponents, grid_values)

    def val_diff_fct():
        x = rnd_points(shape=m)
        coeffs_newton = rnd_points(shape=N)
        v1 = recursive_eval(x, coeffs_newton)
        v2 = iterative_eval(x, coeffs_newton)
        return v1 - v2

    def test_result_equality():
        poly_val_recursive = recursive_eval()
        poly_val_iterative = iterative_eval()
        np.testing.assert_almost_equal(poly_val_recursive, poly_val_iterative)

    test_result_equality()

    if DO_SPEED_TESTS:
        a = recursive_eval
        b = iterative_eval
        print(f'\ntiming {NR_SPEED_SAMPLES} independent runs of each evaluation...')
        time_speedup(a, b)

    if DO_NUMERICAL_TESTS:
        print(f'evaluating numerical error of {NR_NUMERICAL_SAMPLES} independent poly evaluations:')
        test_num_stability(val_diff_fct)

    # test multiple points
    nr_points = NR_SPEED_SAMPLES
    x = np.array([rnd_points(shape=m) for _ in range(nr_points)]).T
    test_result_equality()

    if DO_SPEED_TESTS:
        print(f'\ntiming 1 run of "chained" evaluations (length {NR_SPEED_SAMPLES})...')
        time_speedup(a, b, nr_runs=1)

    # test multiple polynomials (coefficients):
    nr_polynomials = NR_SPEED_SAMPLES
    x = rnd_points(shape=m)
    coeffs_newton = np.array([rnd_points(shape=N) for _ in range(nr_polynomials)]).T

    test_result_equality()

    if DO_SPEED_TESTS:
        print(f'\ntiming {NR_SPEED_SAMPLES} independent runs '
              f'of evaluating {nr_polynomials} polynomials at the same time...')
        time_speedup(a, b)


def test_multiple_coeff_eval(m, n):
    print('test evaluating multiple polynomials (coefficients) at the same time...')
    # NOTE: evaluating the Lagrange monomials on the corresponding interpolation grid should yield the identity matrix
    # NOTE: implemented in regressor.eval_lagrange_monomials_on(points), tested in regression_test.py
    transformer = Transformer(m, n)
    tree = transformer.tree
    points = tree.grid_points
    exponents = tree.gamma
    grid_values = tree.grid_values
    coeffs_newton = transformer.lagrange2newton
    monomial_vals = newt_eval(points, coeffs_newton, exponents, grid_values)
    np.testing.assert_allclose(monomial_vals, np.eye(transformer.N), rtol=RTOL, atol=ATOL)
    # TODO speed benchmarks?!
    # NOTE: recursive implementation does not support the evaluation
    # of multiple coefficients by default! loops slows approach down. unfair comparison


def test_different_settings(test_fct):
    for m, n in product(range(MIN_DIMENSION, MAX_DIMENSION + 1),
                        range(MIN_DEGREE, MAX_DEGREE + 1, 1)):
        assert m > 1, "m must be larger than 1"  # TODO should work for m==1!
        print(f'\n\ndim {m}, degree {n}')
        test_fct(m, n)
    print('... OK.\n\n')


# TODO unit tests
# TODO test numerical error: generate precise ground truths with an increased precision (float 128) and compare
if __name__ == '__main__':
    FUNCTIONS2TEST = [
        test_equality2recursive_eval,  # includes speed benchmarks
        test_multiple_coeff_eval,
    ]
    for fct in FUNCTIONS2TEST:
        test_different_settings(fct)
    print('SUCCESS. all tests passed!')
