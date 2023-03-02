# -*- coding:utf-8 -*-

import numpy as np
import scipy
from numba import f8, njit, void

f = f8
f_list_1D = f[:]
f_list_2D = f[:, :]


def l_u_decompose(matrix):
    # TODO use permutation matrix P
    # P, L, U
    _, L, U = scipy.linalg.lu(matrix)
    # rank of inverse should be maximal (=max_rank)
    U_rank = np.linalg.matrix_rank(U)
    max_rank = min(matrix.shape)
    U_rank_deficiency = max_rank - U_rank
    return L, U, U_rank, U_rank_deficiency


def is_stable_transform(R):
    # check if the data points (<-> transformation R) are not distinct enough
    # important for a numerically stable transformation S
    _, _, _, rank_deficiency = l_u_decompose(R)
    # is_stable = condition_nr <= CONDITION_NR_THRES
    is_stable = rank_deficiency == 0
    return is_stable


def l_u_invert(matrix, verbose: bool = False):
    N_min = min(matrix.shape)
    N_max = max(matrix.shape)
    L1, U1, U1_rank, U1_rank_deficiency = l_u_decompose(matrix)
    if verbose:
        print(f'rank(U1)={U1_rank}; N_min={N_min}; U1 rank deficiency: {U1_rank_deficiency}')
    is_unstable = U1_rank_deficiency > 0
    if is_unstable:
        if verbose:
            print('rank deficiency detected.')
        msg = 'numerical instability detected: matrix U1 has rank deficiencies. the data points are not distinct ' \
              'enough for a stable mapping to the unisolvent interpolation grid nodes.'
        raise np.linalg.LinAlgError(msg)

    # L1_rank = np.linalg.matrix_rank(L1)
    # L1_rank_deficiency = N_min - L1_rank
    # print(f'rank(L1)={L1_rank}; N_min={N_min}; L1 rank deficiency: {L1_rank_deficiency}')
    # TODO resort if necessary. then crop
    # numpy.sort ndarray.sort (apply same sorting to other arrays!)
    # assert U1_rank_deficiency == 0
    T = scipy.linalg.solve(L1[0:N_min, 0:N_min], np.eye(N_min, N_min))
    # U2 = np.eye(N_data, N_data)
    # U1: N_data, N_fit
    # previously:
    # W = scipy.linalg.solve(U1, np.eye(N_min, N_max))
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq
    W, _, _, _ = scipy.linalg.lstsq(U1, np.eye(N_min, N_max))
    W_cropped = W[0:N_max, 0:N_min]
    S1 = W_cropped @ T
    # SC1 = P1 @ R_lu @ S1
    return S1


def invert(R, verbose: bool = False):
    # # Moore-Penrose or pseudo inverse:
    # # using SVD composition
    # # https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
    # S = np.linalg.pinv(R)

    # # by solving the system of equations:
    # S, _, _, _ = scipy.linalg.lstsq(R, np.eye(R.shape[0]))

    # by LU decomposition:
    # ATTENTION: shape (N_min, N_min) depending on numerical properties
    # of transformation matrix R (non invertible L U decomposition)
    S = l_u_invert(R, verbose)
    return S


@njit(f(f_list_1D), cache=True)
def length(x):
    """ computes the length of x

    = global definition of length measurement for higher dimensional space

    NOTE: curse of dimensionality: volume of the domain [-1;1]^m is growing exponentially
    -> space gets harder to search. more and more "distinct" points
    (this should be reflected in the similarity measurement of points)
    "length"/ point similarity considerations for higher dimensions:
    domain diagonal is much "longer" than the domain axes (in terms of l2-norm)
    but l-infinity norm (max-norm) would treat them equally -> introduces bias
    l2-norm of the diagonal actually converges to the l1-norm for m -> infinity
    however the l1-norm is much faster to compute (good compromise)
    NOTE: np.linalg.norm(x, ord=1) computes sum(abs(x)**ord)**(1./ord) <- expensive
    NOTE: also works for matrices
    """
    return np.sum(np.abs(x))  # l_1-norm


@njit(void(f_list_2D, f_list_2D), cache=True)
def fill_distance_matrix(points, distance_matrix):
    nr_points = points.shape[0]
    for i in range(nr_points):  # equal to combinations(range(nr_points), 2)
        point1 = points[i]
        for j in range(i + 1, nr_points):
            point2 = points[j]
            distance = length(point1 - point2)
            distance_matrix[i, j] = distance
            # distances are symmetrical
            distance_matrix[j, i] = distance


def get_distance_matrix(points):
    # computes all inter point similarities
    nr_points = points.shape[0]
    # NOTE: each point is completely similar to itself
    # -> initialise the diagonal with 1
    distance_matrix = np.eye(nr_points)
    fill_distance_matrix(points, distance_matrix)
    return distance_matrix


def make_stable(R, points, values, condition_nr_thres: float = 1.0):
    assert condition_nr_thres >= 1.0, 'the condition number cannot be lower than 1.0'
    condition_nr = np.linalg.cond(R)
    nr_removed_pts = 0
    if condition_nr <= condition_nr_thres:
        return R, points, values, condition_nr, nr_removed_pts

    nr_points = len(values)
    remaining_indices = np.arange(nr_points, dtype=int)
    removed_pts_idxs = []
    distance_matrix = get_distance_matrix(points)

    # remove points until the condition number is low enough:
    while condition_nr > condition_nr_thres:
        # find the closest points (<-> most redundancy)
        idx1, idx2 = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)
        # delete the one with a higher value (minimisation problems)
        if values[idx1] > values[idx2]:
            idx2delete = idx1
        else:
            idx2delete = idx2
        nr_removed_pts += 1
        removed_pts_idxs.append(remaining_indices[idx2delete])
        values = np.delete(values, idx2delete)
        remaining_indices = np.delete(remaining_indices, idx2delete)
        points = np.delete(points, idx2delete, axis=0)
        R = np.delete(R, idx2delete, axis=1)
        nr_remaining_pts = nr_points - nr_removed_pts
        distance_matrix = np.delete(distance_matrix, idx2delete, axis=0)
        distance_matrix = np.delete(distance_matrix, idx2delete, axis=1)

        # TODO
        assert R.shape[0] == nr_remaining_pts
        assert points.shape[1] == nr_remaining_pts
        assert distance_matrix.shape == (nr_remaining_pts, nr_remaining_pts)

    return R, points, values, condition_nr, removed_pts_idxs
