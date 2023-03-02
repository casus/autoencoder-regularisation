# -*- coding:utf-8 -*-
from sys import stdout

import numpy as np
import scipy

from minterpy.transformation import Transformer


# TODO superfluous because of regression?
# TODO refactor API. avoid construction of Transformer object for every call etc.
# NOTE: works only for the same amount of grid points and sample (data) points
def interpolate_scattered(interpol_nodes_scattered, function_values, n, lp_deg):
    """
    use multivariate Newton interpolation [Hecht] on scattered data points
    basic idea: Transformer the Lagrange coefficients for scattered data
        to the Lagrange coefficients on Newton grid distributed data using a change of basis.
    m: dimension
    N: amount of interpolation nodes
    @param interpol_nodes_scattered: 2D (mxN) array of the interpolation nodes (pairwise different!)
    @param function_values: N values of the function to be interpolated at each interpolation node in the same odering
    @param n: interpolation degree
    @param lp_deg: lp_degree-degree of the interpolation polynomial
    @return: lagrange coefficients, newton coefficients, conversion exponents
    """
    # the Lagrange coefficients are equal to the function value at the respective interpolation node
    # -> evaluate the function at each node
    coeffs_lagrange_scattered = function_values.copy()

    # Fit polynomial on scattered data:
    # We now assume that our data can be arbitrarily distributed.

    m, nr_nodes = interpol_nodes_scattered.shape
    transformer = Transformer(m, n, lp_deg)
    assert transformer.N == nr_nodes, 'the amount of interpolation nodes and does not fit the requirements ' \
                                      'with the given interpolation settings'
    # the conversion exponents is symmetrical
    # the same amount of interpolation nodes in both bases (no regression)
    grid2scattered = np.zeros([nr_nodes, nr_nodes])
    e_beta = np.zeros(nr_nodes)
    # Compute transformation ``R`` from scattered grid of Lagrange polynomial
    # to Lagrange Polynomial on Chebyshev grid
    # interpolation based on Michael's multivariate Horner scheme @ Newton coefficients
    print('building transformation matrix:')
    raise DeprecationWarning('simplifications possible. iterative evaluation is able to handle multiple points')
    nr_matrix_entries = nr_nodes ** 2
    ctr = 0
    for j in range(nr_nodes):
        D_alpha = transformer.lagrange2newton[:, j]
        for i in range(nr_nodes):
            e_beta[i] = transformer.tree.eval(interpol_nodes_scattered[:, i], D_alpha)
            ctr += 1
        grid2scattered[:, j] = e_beta
        stdout.write(f'progress: {ctr}/{nr_matrix_entries} entries ({ctr / nr_matrix_entries:.2%})\r')
        stdout.flush()

    stdout.write("\n")  # move the cursor to the next line
    scattered2grid = scipy.linalg.inv(grid2scattered)  # S = R^{-1}
    # Project scattered lagrange coefficients  to Newton grid coefficients
    coeffs_lagrange = np.dot(scattered2grid, coeffs_lagrange_scattered)
    # TODO
    # convert into Newton form
    coeffs_newton = transformer.transform_l2c(coeffs_lagrange)
    return coeffs_newton, coeffs_lagrange, scattered2grid
