from itertools import product

from minterpy.derivation import *
from minterpy.transformation import Transformer
from minterpy.utils import assert_shape, assert_all_equal

np.random.seed(42)

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


def get_rnd_parameters(*args):
    return 2 * (np.random.rand(*args) - 0.5)


def test_canonical_derivation():
    print('\ntesting derivation...')
    # ATTENTION: the exponent vectors of all derivatives have to be included already!
    exponents = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 2]]).T
    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert exponents.shape == (3, 5)
    assert coeffs.shape == (5,)

    coeffs_canonical_deriv = partial_derivative_canonical(0, coeffs, exponents)
    # coefficients should become 0.0
    assert np.allclose(coeffs_canonical_deriv, np.array([0.0, 0.0, 0.0, 0.0, 0.0])), \
        f"unexpected coefficients {coeffs_canonical_deriv}"

    coeffs_canonical_deriv = partial_derivative_canonical(1, coeffs, exponents)
    # coefficients should "change places"
    assert np.allclose(coeffs_canonical_deriv, np.array([2.0, 0.0, 4.0, 0.0, 0.0])), \
        f"unexpected coefficients {coeffs_canonical_deriv}"

    coeffs_canonical_deriv = partial_derivative_canonical(2, coeffs, exponents)
    # NOTE: coefficients should be multiplied with the exponent
    assert np.allclose(coeffs_canonical_deriv, np.array([3.0, 4.0, 10.0, 0.0, 0.0])), \
        f"unexpected coefficients {coeffs_canonical_deriv}"
    print('tests passed!')


def test_canonical_gradient():
    print('\ntesting gradient construction...')
    # ATTENTION: the exponent vectors of all derivatives have to be included already!
    exponents = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 2]]).T
    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert exponents.shape == (3, 5)
    assert coeffs.shape == (5,)

    grad = derive_gradient_canonical(coeffs, exponents)
    grad_expected = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                              [2.0, 0.0, 4.0, 0.0, 0.0],
                              [3.0, 4.0, 10.0, 0.0, 0.0]])
    assert grad.shape == exponents.shape, f'unexpected gradient shape: {grad.shape}'
    assert np.allclose(grad, grad_expected), f"unexpected gradient: {grad}"
    print('tests passed!')


def test_lagrange_gradient(m, n):
    print('testing gradient operator tensor construction...')
    # define a random Lagrange polynomial
    transformer = Transformer(m, n)
    N = transformer.N
    coeffs_lagrange = get_rnd_parameters(N)  # [-1;1]
    coeffs_canonical = transformer.transform_l2c(coeffs_lagrange)
    expected_gradient_shape = (m, N)
    expected_gradient_operator_shape = (m, N, N)
    l2c = transformer.lagrange2canonical
    c2l = transformer.canonical2lagrange
    exponents = transformer.exponents

    grad2_c2c = get_canonical_gradient_operator(exponents)

    identity = np.eye(N)
    c2c = identity
    grad = get_gradient_operator(c2c, c2c, exponents)
    np.testing.assert_equal(grad, grad2_c2c)

    grad1_l2l = get_lagrange_gradient_operator(l2c, c2l, exponents)
    # convert precomputed canonical operator to Lagrange operator
    grad2_l2c = tensor_right_product(grad2_c2c, l2c)
    grad2_l2l = tensor_left_product(c2l, grad2_l2c)
    assert_shape([grad1_l2l, grad2_l2l, grad2_c2c], expected_gradient_operator_shape)
    assert_all_equal([grad1_l2l, grad2_l2l])

    grad1_lagrange = get_gradient(coeffs_lagrange, grad1_l2l)
    grad1_canonical = np.apply_along_axis(transformer.transform_l2c, 1, grad1_lagrange)

    # compute with precomputed canonical operator
    grad2_canonical = get_gradient(coeffs_canonical, grad2_c2c)
    grad2_lagrange = np.apply_along_axis(transformer.transform_c2l, 1, grad2_canonical)

    # derive without precomputed operator
    grad3_canonical = derive_gradient_canonical(coeffs_canonical, exponents)
    grad3_lagrange = np.apply_along_axis(transformer.transform_c2l, 1, grad3_canonical)

    # compute with precomputed canonical operator converted to Lagrange operator
    grad4_lagrange = get_gradient(coeffs_lagrange, grad2_l2l)
    grad4_canonical = np.apply_along_axis(transformer.transform_l2c, 1, grad4_lagrange)

    assert_shape([grad1_canonical, grad1_lagrange, grad2_canonical, grad2_lagrange, grad3_canonical, grad3_lagrange,
                  grad4_canonical, grad4_lagrange], expected_gradient_shape)
    assert_all_equal([
        grad1_canonical,
        grad2_canonical,
        grad3_canonical,
        grad4_canonical])
    assert_all_equal([
        grad1_lagrange,
        grad2_lagrange,
        grad3_lagrange,
        grad4_lagrange])

    # the evaluation of the gradient in Newton basis on the grid points should yield the Lagrange gradient
    # NOTE: serves as a test for the Newton evaluation fct
    l2n = transformer.lagrange2newton
    grad_op_l2n = tensor_left_product(l2n, grad1_l2l)  # (m x N x N) tensor
    gradient_newton = get_gradient(coeffs_lagrange, grad_op_l2n)  # (m x N)
    tree = transformer.tree
    gradient_vals = tree.eval(tree.grid_points, gradient_newton.T).T
    assert_all_equal([gradient_vals, grad1_lagrange])


def test_gradient_analytical(m, n):
    if n != 2:
        print('only working with n==2. skipping tests')
        return
    print('testing gradient with analytical example...')

    # define a (scalar) quadratic function for which we already know the derivative (=gradient)
    n = 2
    transformer = Transformer(m, n)
    a = get_rnd_parameters(m)
    A = np.diag(a)
    b = get_rnd_parameters(1, m)
    c = get_rnd_parameters(1)
    f = lambda x: x.T @ A @ x + b @ x + c
    fx = lambda x: 2 * x.T @ A + b
    # evaluate f as well as its derivative fx at the Chebyshev nodes of our polynomial:
    f_vals = np.apply_along_axis(f, 0, transformer.grid_points)
    fx_vals = np.apply_along_axis(fx, 0, transformer.grid_points).squeeze()
    # compute the gradient of our polynomial at each grid_point (equal to gradient in Lagrange basis)
    l2c = transformer.lagrange2canonical
    c2l = transformer.canonical2lagrange
    grad_op_l2l = get_lagrange_gradient_operator(l2c, c2l,
                                                 transformer.exponents)
    coeffs_lagrange = f_vals.reshape(-1)
    # the lagrange coefficients of the gradient (partial derivatives) are the polynomial values
    # at the corresponding interpolation nodes
    grad_lagrange = get_gradient(coeffs_lagrange, grad_op_l2l)
    # should be equal to the values of the analytical derivative
    assert_all_equal([fx_vals, grad_lagrange])


def test_different_settings(test_fct):
    for m, n in product(range(MIN_DIMENSION, MAX_DIMENSION + 1),
                        range(MIN_DEGREE, MAX_DEGREE + 1, 1)):
        assert m > 1, "m must be larger than 1"  # TODO should work for m==1!
        print(f'\ndim {m}, degree {n}')
        test_fct(m, n)
    print('... OK.\n\n')


# TODO into unit tests, test suite
if __name__ == '__main__':

    test_canonical_derivation()
    test_canonical_gradient()

    FUNCTIONS2TEST = [
        test_lagrange_gradient,
        test_gradient_analytical,
    ]
    for fct in FUNCTIONS2TEST:
        test_different_settings(fct)

    print('SUCCESS. all tests passed!')
