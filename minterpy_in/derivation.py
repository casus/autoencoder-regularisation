# -*- coding:utf-8 -*-
from typing import Optional

import numpy as np

from minterpy.transformation import Transformer


# TODO integrate derivation code directly into Interpolator (or better Interpolant) class
# API: e.g. grad = interpolant.gradient()
# TODO gradient is composed of m polynomials itself. defined on the same grid/tree


# TODO numba just in time compile
#  @jit
def get_match_idx(exponents: np.ndarray, monomial_exponents: np.ndarray) -> int:
    nr_dimensions, nr_monomials = exponents.shape
    for coeff_idx in range(nr_monomials):
        for dim_idx in range(nr_dimensions):
            if exponents[dim_idx, coeff_idx] != monomial_exponents[dim_idx]:
                break  # check next monomial_exponents
            elif dim_idx == nr_dimensions - 1:  # last element has matched
                return coeff_idx
    raise ValueError(f"exponents of derivative {monomial_exponents} are not present in the exponent matrix")


def partial_derivative_canonical(dim_idx: int, coeffs_canonical: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """
    :param dim_idx: the index of the dimension to derive with respect to
    :param coeffs_canonical: the coefficients of the polynomial in canonical form
    :param exponents: the respective exponent vectors of all monomials
    :return: the coefficients of the partial derivative in the same ordering (and canonical form)
    """
    coeffs_canonical_deriv = np.zeros(coeffs_canonical.shape)
    for monomial_idx, coeff in enumerate(coeffs_canonical):
        monomial_exponents = exponents[:, monomial_idx]
        if monomial_exponents[dim_idx] > 0:
            mon_exponents_derived = monomial_exponents.copy()
            mon_exponents_derived[dim_idx] -= 1
            # "gradient exponential mapping"
            new_coeff_idx = get_match_idx(exponents, mon_exponents_derived)
            # multiply with exponent
            coeffs_canonical_deriv[new_coeff_idx] = coeff * exponents[dim_idx, monomial_idx]
    return coeffs_canonical_deriv


def derive_gradient_canonical(coeffs_canonical: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """ derives the gradient without any precomputation

    :param coeffs_canonical: the coefficients of the polynomial in canonical form
    :param exponents: the respective exponent vectors of all monomials
    :return: the gradient in canonical form
    """
    dimensionality, nr_monomials = exponents.shape
    nr_coefficients = len(coeffs_canonical)
    assert nr_monomials == nr_coefficients, 'coefficient and exponent shapes do not match: ' \
                                            f'{coeffs_canonical.shape}, {exponents.shape}'
    gradient = np.empty((dimensionality, nr_monomials))
    for dim_idx in range(dimensionality):
        coeffs_canonical_deriv = partial_derivative_canonical(dim_idx, coeffs_canonical, exponents)
        gradient[dim_idx, :] = coeffs_canonical_deriv
    return gradient


def partial_derivative_lagrange(transformer: Transformer, dim_idx: int,
                                coeffs_lagrange: np.ndarray) -> np.ndarray:
    coeffs_canonical = transformer.transform_l2c(coeffs_lagrange)
    # exponents are equal to the multi indices of the interpolation tree
    exponents = transformer.tree.exponents
    coeffs_canonical_deriv = partial_derivative_canonical(dim_idx, coeffs_canonical, exponents)
    coeffs_lagrange_deriv = transformer.transform_c2l(coeffs_canonical_deriv)
    return coeffs_lagrange_deriv


def gradient_lagrange(transformer: Transformer, coeffs_lagrange: np.ndarray) -> np.ndarray:
    assert len(coeffs_lagrange) == transformer.N
    dimensionality = transformer.m
    coeffs_canonical = transformer.transform_l2c(coeffs_lagrange)
    # exponents are equal to the multi indices of the interpolation tree
    exponents = transformer.tree.exponents
    grad_canonical = derive_gradient_canonical(coeffs_canonical, exponents)
    grad_lagrange = np.empty(grad_canonical.shape)
    for dim_idx in range(dimensionality):
        grad_lagrange[dim_idx, :] = transformer.transform_c2l(grad_canonical[dim_idx, :])
    return grad_lagrange


def assert_is_square(a: np.ndarray, size: Optional[int] = None, specifier: str = 'matrix'):
    shape = a.shape
    if len(shape) != 2:
        raise ValueError(f'{specifier} should be 2D')
    if shape[0] != shape[1]:
        raise ValueError(f'{specifier} is not square: {shape}')
    if size is not None and shape[0] != size:
        raise ValueError(f'{specifier} should be of size {size} but is of size {shape[0]}')


def get_canonical_gradient_operator(exponents: np.ndarray) -> np.ndarray:
    """ constructs the tensor transforming canonical coefficients into the canonical coefficients of the gradient

    NOTE: the gradient operator in canonical basis is sparse!
    -> obtaining the gradient operator for different bases by matrix multiplications with the transformation matrices
        is inefficient

    @param exponents: matrix of all exponent vectors. has to contain also the exponent vectors of all derivatives,
     which can otherwise not be found
    @return: the gradient operation tensor from canonical basis to canonical basis (<- sparse!)
    """
    dimensionality, nr_monomials = exponents.shape

    grad_c2c = np.zeros((dimensionality, nr_monomials, nr_monomials))
    for coeff_idx_from in range(nr_monomials):
        monomial_exponents = exponents[:, coeff_idx_from]
        for dim_idx in range(dimensionality):  # derivation in every dimension
            monomial_exponent_dim = monomial_exponents[dim_idx]
            if monomial_exponent_dim > 0:
                mon_exponents_derived = monomial_exponents.copy()
                mon_exponents_derived[dim_idx] -= 1
                # "gradient exponential mapping":
                # determine where each coefficient gets mapped to
                # -> the idx of the derivative monomial
                coeff_idx_to = get_match_idx(exponents, mon_exponents_derived)
                # also multiply with exponent
                grad_c2c[dim_idx, coeff_idx_to, coeff_idx_from] = monomial_exponent_dim
    return grad_c2c


def tensor_right_product(tensor: np.ndarray, right_factor: np.ndarray):
    # tensor dot product a . b: sum over the last axis of a and the first axis of b in order
    # = sum over tensor axis 2 and right factor axis 0
    # (m x N x N) @ (N x k) -> (m x N x k)
    return np.tensordot(tensor, right_factor, axes=1)  # <- dot product


def get_gradient(coefficients, gradient_operator):
    """ computes the gradient using a precomputed operator tensor

    @param coefficients: the coefficients of a polynomial in basis a
    @param gradient_operator: the gradient operation tensor from basis a to b
    @return: the gradient in basis b
    """
    return tensor_right_product(gradient_operator, coefficients)


def tensor_left_product(left_factor: np.ndarray, tensor: np.ndarray):
    """

    in the case of multiplication with e.g. a conversion matrix (N x X):
    tensor should maintain shape: (N x N) @ (m x N x N) -> (m x N x N)


    """
    # TODO remove. legacy:
    # tensor = np.swapaxes(tensor, 1, 2)
    # product = tensor_right_product(tensor, left_factor.T)
    # # "transpose" back:
    # product = np.swapaxes(product, 1, 2)

    tensor = np.transpose(tensor, (1, 0, 2))  # (N x m x N)
    product = np.tensordot(left_factor, tensor, axes=1)  # <- dot product
    # "transpose" back:
    product = np.transpose(product, (1, 0, 2))  # (N x m x N)
    return product


# TODO cache (pickle) for increased speed
# TODO numba just in time compile
#  @jit
def get_gradient_operator(x2c: np.ndarray, c2x: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """ computes the gradient operation tensor from a variable basis to a variable basis

    NOTE: the gradient operator in canonical basis is sparse!
    directly computing the tensor with the transformation is more efficient than obtaining the gradient operator
        by matrix multiplications with the transformation matrices!

    @param x2c: transformation matrix from basis x to canonical basis
    @param c2x: transformation matrix from canonical basis to basis x
    @param exponents: matrix of all exponent vectors.
        ATTENTION: has to contain also the exponent vectors of all derivatives, which can otherwise not be found
            <-> exponents = multi indices have to be "complete"!
    @return: the tensor of the m partial derivative operators from Lagrange basis to Lagrange basis
    """
    dimensionality, nr_monomials = exponents.shape
    assert_is_square(x2c, size=nr_monomials)
    assert_is_square(c2x, size=nr_monomials)

    # gradient operation tensor from Lagrange basis to canonical basis
    grad_x2c = np.zeros((dimensionality, nr_monomials, nr_monomials))
    for coeff_idx_from in range(nr_monomials):
        monomial_exponents = exponents[:, coeff_idx_from]
        for dim_idx in range(dimensionality):  # derivation in every dimension
            monomial_exponent_dim = monomial_exponents[dim_idx]
            if monomial_exponent_dim > 0:  # use sparsity and avoid unnecessary operations
                mon_exponents_derived = monomial_exponents.copy()
                mon_exponents_derived[dim_idx] -= 1
                # "gradient exponential mapping":
                # determine where each coefficient gets mapped to
                # -> the idx of the derivative monomial
                coeff_idx_to = get_match_idx(exponents, mon_exponents_derived)
                # also multiply with exponent (cf. with canonical case: get_canonical_gradient_operator() )
                # Lagrange:
                # matrix multiplication: C = A @ B
                # -> C[i,:] += A[i,j] * B[j,:] ("from j to i")
                # NOTE: addition required!
                # gradient_lagrange = gradient_canonical @ l2c
                grad_x2c[dim_idx, coeff_idx_to, :] += monomial_exponent_dim * x2c[coeff_idx_from, :]

    # back to Lagrange basis
    grad_x2x = tensor_left_product(c2x, grad_x2c)

    return grad_x2x


def get_lagrange_gradient_operator(l2c: np.ndarray, c2l: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """ computes the gradient operation tensor from Lagrange basis to Lagrange basis

    NOTE: the gradient operator in canonical basis is sparse!
    directly computing the tensor with the transformation is more efficient than obtaining the gradient operator
        by matrix multiplications with the transformation matrices!

    @param l2c: transformation matrix from Lagrange to canonical basis
    @param c2l: transformation matrix from canonical to Lagrange basis
    @param exponents: matrix of all exponent vectors. has to contain also the exponent vectors of all derivatives,
     which can otherwise not be found
    @return: the tensor of the m partial derivative operators from Lagrange basis to Lagrange basis
    """
    return get_gradient_operator(l2c, c2l, exponents)
