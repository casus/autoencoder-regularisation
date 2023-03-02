# -*- coding:utf-8 -*-

import time
from itertools import product

import numpy as np
import scipy

from minterpy.scattered_data import interpolate_scattered
from minterpy.transformation import Transformer
from minterpy.unit_test import RUNGE_FCT_VECTORIZED
from minterpy.utils import apply_vectorized, get_eval_fct, get_eval_fct_canonical, rnd_points

MIN_DEGREE = 1
MAX_DEGREE = 3

ONLY_UNEVEN_DEGREES = True  # test only uneven interpolation total_degrees (result in more symmetrical grid)
if ONLY_UNEVEN_DEGREES:
    assert MIN_DEGREE % 2 == 1
    assert MAX_DEGREE % 2 == 1
    DEGREE_STEP = 2
else:
    DEGREE_STEP = 1

MIN_DIMENSION = 2
MAX_DIMENSION = 3

TEST_RUN_SETTINGS = product(range(MIN_DIMENSION, MAX_DIMENSION + 1), range(MIN_DEGREE, MAX_DEGREE + 1, DEGREE_STEP))

LP_DEGREE = 2

PRINT_RESULTS_SINGLE_RUN = False

EXP_FMT_STR = '{:.2e}'
NR_RUNS_DEFAULT = 1


# TODO integrate into tree class, inverse of existing Transformer
def transform_c2n(transformer, coeffs_canonical):
    conv_matrix_c2n = scipy.linalg.inv(transformer.newton2canonical)
    return np.dot(conv_matrix_c2n, coeffs_canonical)


def interpolate_naive(test_tr, interpol_nodes_scattered, node_values_groundtruth):
    # naive interpolation by inverting the Vandermonde exponents
    vandermonde_matrix = test_tr.buildCanonicalVandermonde(interpol_nodes_scattered.T, test_tr.exponents)
    vandermonde_matrix_inv = scipy.linalg.inv(vandermonde_matrix)
    coeffs_canonical_naive = np.dot(vandermonde_matrix_inv, node_values_groundtruth)
    return coeffs_canonical_naive


def extract_key_metrics(coeffs_canonical, coeffs_canonical_naive, coeffs_canonical_truth,
                        node_vals, node_vals_naive, node_vals_truth):
    coeff_avg_abs_dev = None
    coeff_avg_abs_dev_naive = None
    if coeffs_canonical_truth is not None:
        coeff_avg_abs_dev = np.mean(np.abs(coeffs_canonical_truth - coeffs_canonical))
        coeff_avg_abs_dev_naive = np.mean(np.abs(coeffs_canonical_truth - coeffs_canonical_naive))

    val_avg_abs_dev = np.mean(np.abs(node_vals_truth - node_vals))
    val_avg_abs_dev_naive = np.mean(np.abs(node_vals_truth - node_vals_naive))

    def report_difference(diff, description=''):
        # TODO more elaborate reporting
        # TODO plot
        print('\n')
        print(description)
        # print(diff)
        diff_abs = np.abs(diff)
        print('min abs:', EXP_FMT_STR.format(np.min(diff_abs)))
        print('mean abs:', EXP_FMT_STR.format(np.mean(diff_abs)))
        print('max abs:', EXP_FMT_STR.format(np.max(diff_abs)))
        # print('\n')

    # compare coefficients
    # NOTE: conversion into the different bases with the conversion matrices of a transformer_from object
    #   are defined for polynomials interpolated on newton grids
    # conversion can hence not be used for the polynomial obtained by the naive interpolation approach on scattered data

    if PRINT_RESULTS_SINGLE_RUN:
        coeffs_diff_naive = coeffs_canonical_truth - coeffs_canonical_naive
        description = 'coefficient deviation, Vandermonde exponents inversion:'
        report_difference(coeffs_diff_naive, description)

        coeffs_diff = coeffs_canonical_truth - coeffs_canonical
        description = 'coefficient deviation, Multivariate Newton Interpolation:'
        report_difference(coeffs_diff, description)

        node_vals_diff_naive = node_vals_truth - node_vals_naive
        description = 'function value difference on scattered interpolation nodes, Vandermonde exponents inversion:'
        report_difference(node_vals_diff_naive, description)

        node_vals_diff = node_vals_truth - node_vals
        description = 'function value difference on scattered interpolation nodes, Multivariate Newton Interpolation:'
        report_difference(node_vals_diff, description)

    return coeff_avg_abs_dev, coeff_avg_abs_dev_naive, val_avg_abs_dev, val_avg_abs_dev_naive


def single_run(transformer_object, g_vectorized, coeffs_canonical_truth, m, n):
    num_nodes_scattered = transformer_object.N
    lp_deg = transformer_object.lp_degree
    interpol_nodes_scattered = 2 * (np.random.rand(m, num_nodes_scattered) - 0.5)  # scattered points
    # visualize newton nodes (orange) vs. the scattered points (blue).
    # plt.scatter(interpol_nodes_scattered[0, :], interpol_nodes_scattered[1, :])
    # plt.scatter(interpol_nodes[0, :], interpol_nodes[1, :])
    t1 = time.time()
    node_vals_truth = g_vectorized(interpol_nodes_scattered)
    if PRINT_RESULTS_SINGLE_RUN:
        print("\ngroundtruth generated in %.2fs" % (time.time() - t1))
    t2 = time.time()
    coeffs_newton, coeffs_lagrange, conv_scattered2grid = interpolate_scattered(interpol_nodes_scattered,
                                                                                node_vals_truth, n, lp_deg)
    if PRINT_RESULTS_SINGLE_RUN:
        print("interpolation runtime: %.2f s" % (time.time() - t2))
    coeffs_canonical = transformer_object.transform_n2c(coeffs_newton)
    coeffs_canonical_naive = interpolate_naive(transformer_object, interpol_nodes_scattered, node_vals_truth)
    # compare the function values of the two interpolated polynomials at the scattered interpolation nodes
    eval_newton_poly = get_eval_fct(transformer_object.tree, coeffs_newton, m, n, num_nodes_scattered,
                                    transformer_object.gamma_placeholder)
    node_vals_interpol = apply_vectorized(eval_newton_poly, interpol_nodes_scattered)

    exponents = transformer_object.exponents
    eval_fct_naive = get_eval_fct_canonical(coeffs_canonical_naive, exponents)
    node_vals_naive = apply_vectorized(eval_fct_naive, interpol_nodes_scattered)
    return extract_key_metrics(coeffs_canonical, coeffs_canonical_naive, coeffs_canonical_truth, node_vals_interpol,
                               node_vals_naive, node_vals_truth)


def summarise_tuple_list(list_of_tuples):
    result = []
    for x in zip(*list_of_tuples):
        if len(x) > 0 and x[0] is not None:
            result.append(sum(x) / len(x))
        else:
            result.append(None)
    return result


def test_fct_scattered(transformer_object, g_vectorized, coeffs_canonical_truth, m, n, lp_deg=2,
                       nr_runs=NR_RUNS_DEFAULT):
    # compare multivariate Newton interpolation [Hecht]
    # to naive interpolation by inverting the Vandermonde exponents
    # on scattered data

    results = [single_run(transformer_object, g_vectorized, coeffs_canonical_truth, m, n) for i in
               range(nr_runs)]

    return summarise_tuple_list(results)
    # coeff_avg_abs_dev, coeff_avg_abs_dev_naive, val_avg_abs_dev, val_avg_abs_dev_naive


def test_runge_fct(nr_runs=NR_RUNS_DEFAULT):
    print('comparing Multivariate Newton Interpolation to naive interpolation by inverting the Vandermonde exponents\n'
          'on scattered interpolation nodes\n'
          'interpolating a runge function')
    lp_deg = LP_DEGREE
    for m, n in TEST_RUN_SETTINGS:
        print(f'\n\ncomparison with settings:\nm = {m}, n = {n}, lp-Degree = {lp_deg}\naveraging {nr_runs} runs...')
        result = test_fct_scattered(RUNGE_FCT_VECTORIZED, coeffs_truth, m, n, lp_deg, nr_runs)
        coeff_avg_abs_dev, coeff_avg_abs_dev_naive, val_avg_abs_dev, val_avg_abs_dev_naive = result
        print('results:')
        print('function value difference on scattered interpolation nodes:')
        print(EXP_FMT_STR.format(val_avg_abs_dev), 'Multivariate Newton Interpolation')
        print(EXP_FMT_STR.format(val_avg_abs_dev_naive), 'Vandermonde exponents inversion')


def test_rnd_poly(nr_runs=NR_RUNS_DEFAULT):
    print('comparing Multivariate Newton Interpolation to naive interpolation by inverting the Vandermonde exponents\n'
          'on scattered interpolation nodes\n'
          'interpolating a random polynomial of the same degree')
    lp_deg = LP_DEGREE
    for m, n in TEST_RUN_SETTINGS:
        print(f'\n\nsettings:\nm = {m}, n = {n}, lp-Degree = {lp_deg}\naveraging {nr_runs} runs...')
        transformer_object = Transformer(m, n, lp_deg)
        num_nodes = transformer_object.N
        # the number of scattered interpolation nodes has to be equal to
        # the number of newton grid interpolation nodes (<- no regression happening)
        num_nodes_scattered = num_nodes
        # random coefficients of a polynomial in canonical form
        coeffs_truth = rnd_points(num_nodes_scattered)
        exponents = transformer_object.exponents
        eval_fct_naive = get_eval_fct_canonical(coeffs_truth, exponents)

        def eval_fct_vectorised(eval_points):
            return np.apply_along_axis(eval_fct_naive, 0, eval_points)

        result = test_fct_scattered(transformer_object, eval_fct_vectorised, coeffs_truth, m, n, nr_runs)
        coeff_avg_abs_dev, coeff_avg_abs_dev_naive, val_avg_abs_dev, val_avg_abs_dev_naive = result
        print('results:')
        if coeff_avg_abs_dev is None:
            raise ValueError
        print('coefficient deviation:')
        print(EXP_FMT_STR.format(coeff_avg_abs_dev), 'Multivariate Newton Interpolation')
        print(EXP_FMT_STR.format(coeff_avg_abs_dev_naive), ' Vandermonde exponents inversion')
        print('function value difference on scattered interpolation nodes:')
        print(EXP_FMT_STR.format(val_avg_abs_dev), 'Multivariate Newton Interpolation')
        print(EXP_FMT_STR.format(val_avg_abs_dev_naive), 'Vandermonde exponents inversion')


if __name__ == '__main__':
    test_rnd_poly()
    test_runge_fct()
