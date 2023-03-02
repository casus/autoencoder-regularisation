# -*- coding:utf-8 -*-

# TODO move to tests folder

import time
import unittest
from argparse import ArgumentParser
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from minterpy.solvers import DDS
from minterpy.transformation import Transformer
# import pytest
# import concurrent.futures
# from mpl_toolkits.mplot3d import Axes3D
# import utils
from minterpy.utils import get_eval_fct, report_error, rnd_points

# TODO into tests/ directory
np.random.seed(42)

DESIRED_PRECISION = 10  # maximal allowed precision
NR_SAMPLE_POINTS = 1000  # TODO dependent on m?

MIN_DEGREE = 1
MAX_DEGREE = 5

ONLY_UNEVEN_DEGREES = True  # test only uneven interpolation total_degrees (result in more symmetrical grid)
if ONLY_UNEVEN_DEGREES:
    assert MIN_DEGREE % 2 == 1
    assert MAX_DEGREE % 2 == 1
    DEGREE_STEP = 2
else:
    DEGREE_STEP = 1

MIN_DIMENSION = 1
MAX_DIMENSION = 4

RUNGE_FCT = lambda x: 1 / (1 + 25 * np.linalg.norm(x) ** 2)

# TODO more
TEST_FUNCTIONS = [  # ground truths:
    RUNGE_FCT,
    # lambda x: np.sin((2 * x[0]) / np.pi + x[1]), # TODO fixed 2D
    # lambda x: x[0] + x[1], # TODO fixed 2D
]

RUNGE_FCT_VECTORIZED = lambda eval_points: np.apply_along_axis(RUNGE_FCT, 0, eval_points)

# TODO
# TEST_FUNCTIONS_VECTORIZED = [lambda eval_points: np.apply_along_axis(g, 0, eval_points) for g in TEST_FUNCTIONS]
TEST_FUNCTIONS_VECTORIZED = [RUNGE_FCT_VECTORIZED]


# TODO more sophisticated tests
# test if interpolation is globally converging
# test grid structure
# test polynomial conversion
# test polynomial evaluation


# TODO more thorough tests


def test_fct_regular(m, n, lp_deg=2.0):
    ground_truth_fct = RUNGE_FCT_VECTORIZED
    print("\n\n  - dimension: %d" % (m))
    print("  - degree: %d" % (n))
    print("  - lp_degree: %d" % (lp_deg))

    t1 = time.time()
    transformer = Transformer(m, n, lp_degree=lp_deg)
    tr = transformer.tree
    # tree.grid_points: interpolation points ("grid")
    print("Tree built in %1.2es" % (time.time() - t1))
    N = transformer.N
    print("  - no. coefficients: %d" % (N))

    # multi indices
    gamma = np.zeros(m, dtype=np.int_)

    fct_values = ground_truth_fct(tr.grid_points)  # evaluate ground truths function
    print("Groundtruth generated in %1.2es" % (time.time() - t1))

    # estimate parameters using Divided Differences Scheme (interpolation)
    t1 = time.time()
    dds = DDS()
    coeffs_newton = dds.run(m, N, tr.tree, fct_values, tr.grid_values, gamma, 1, 1)
    print("DDS (interpolation) took %1.2es" % (time.time() - t1))

    # test equality Transformer.interpolate()
    coeffs_newton2 = transformer.interpolate(fct_values)
    assert np.allclose(coeffs_newton, coeffs_newton2)

    # test transformation
    # the Lagrange coefficients (on the interpolation grid)
    # should be equal to the function values (on the interpolation grid)
    coeffs_lagrange = transformer.transform_n2l(coeffs_newton)
    assert np.allclose(coeffs_lagrange, fct_values)

    eval_newton_poly = get_eval_fct(tr, coeffs_newton)

    """
    Evaluate polynomial at all interpolation points (sanity check)
    error should be at machine precision
    """
    pts_interpolation_nodes = tr.grid_points
    vals_interpol = tr.eval(pts_interpolation_nodes, coeffs_newton)
    err = fct_values - vals_interpol
    report_error(err, 'error at interpolation nodes (accuracy check)')
    np.testing.assert_almost_equal(vals_interpol, fct_values, decimal=DESIRED_PRECISION)

    """
       Evaluate polynomial on uniformly sampled points
    """
    pts_uniformly_random = rnd_points(m, NR_SAMPLE_POINTS)
    vals_interpol = tr.eval(pts_uniformly_random, coeffs_newton)
    vals_true = ground_truth_fct(pts_uniformly_random)
    err = vals_true - vals_interpol
    report_error(err, f'error on {NR_SAMPLE_POINTS} uniformly random points in [-1,1]^m')

    if m != 2:
        return
    """
    Evaluate polynomial on equidistant grid
    """
    # TODO option to specify density!
    x = np.arange(-1, 1, step=0.1)
    y = np.arange(-1, 1, step=0.1)
    x, y = np.meshgrid(x, y)
    equidist_grid = np.stack([x.reshape(-1), y.reshape(-1)])

    vals_interpol = tr.eval(equidist_grid, coeffs_newton)
    vals_true = ground_truth_fct(equidist_grid)
    err = vals_true - vals_interpol

    report_error(err, 'error on equidistant grid')

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(vals_true.reshape([len(x), len(x)]))
    ax1.set_title("ground truths")
    ax2.imshow(vals_interpol.reshape([len(x), len(x)]))
    ax2.set_title("interpolant")
    plt.show()


def test_interpolating_polynomial(m, n, lp_deg=2.0):
    print("\n\n  - dimension: %d" % (m))
    print("  - degree: %d" % (n))
    print("  - lp_degree: %d" % (lp_deg))

    t1 = time.time()
    transformer = Transformer(m, n, lp_degree=lp_deg)
    print("Tree built in %1.2es" % (time.time() - t1))

    N = transformer.N
    print("  - no. coefficients: %d" % (N))

    # generate ground truth polynomial
    coeffs_newton_true = rnd_points(N)  # evaluate ground truths function
    coeffs_lagrange_true = transformer.transform_n2l(coeffs_newton_true)
    # fct_values: the function values that we want to fit using our polynomial
    fct_values = coeffs_lagrange_true

    # NOTE: transformer.interpolate() <- DDS
    # is actually the transformation from Lagrange to Newton basis
    coeffs_newton = transformer.transform_l2n(coeffs_lagrange_true)
    np.testing.assert_almost_equal(coeffs_newton, coeffs_newton_true, decimal=DESIRED_PRECISION)
    err = coeffs_newton - coeffs_newton_true
    report_error(err, f'error of the interpolated Newton coefficients (transformation):')

    # interpolate and then compare coefficients (-> numerical error)
    coeffs_newton = transformer.interpolate(fct_values)
    np.testing.assert_almost_equal(coeffs_newton, coeffs_newton_true, decimal=DESIRED_PRECISION)

    err = coeffs_newton - coeffs_newton_true
    report_error(err, f'error of the interpolated Newton coefficients (DDS):')

    # test transformation
    # the Lagrange coefficients (on the interpolation grid)
    # should be equal to the function values (on the interpolation grid)
    coeffs_lagrange = transformer.transform_n2l(coeffs_newton)
    assert np.allclose(coeffs_lagrange, coeffs_lagrange_true)
    err = coeffs_lagrange - coeffs_lagrange_true
    report_error(err, f'error of the Lagrange coefficients (function values):')


def test_different_settings(test_fct):
    # for g_vectorized in TEST_FUNCTIONS_VECTORIZED:
    for m, n in product(range(MIN_DIMENSION, MAX_DIMENSION + 1),
                        range(MIN_DEGREE, MAX_DEGREE + 1, DEGREE_STEP)):
        test_fct(m, n)


class MainPackageTest(unittest.TestCase):

    # test settings:
    # setting1 = False

    @classmethod
    def setUpClass(cls):
        # preparations which have to be made only once
        pass

    # TEST CASES:
    # NOTE: all test case names have to start with "test..."
    @staticmethod
    def test_correctness():
        print('testing interpolation of ground truth polynomials:')
        test_different_settings(test_interpolating_polynomial)

    # @staticmethod
    # def test_accuracy():
    #     test_different_settings(test_fct_regular)


# TODO tests for scattered data approach


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", dest="m", type=int, help="input dimension", default=2)
    parser.add_argument("-n", dest="n", type=int, help="polynomial degree", default=15)
    parser.add_argument("-lp_degree", dest="lp_degree", type=float, help="LP order", default=2)
    args = parser.parse_args()

    n = args.n
    m = args.m
    lp_deg = args.lp

    assert m > 0, "m must be larger than 0"
    assert m > 1, "m must be larger than 1"  # TODO should work for m==1!
    test_fct_regular(m, n, lp_deg)
