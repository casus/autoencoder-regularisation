# -*- coding:utf-8 -*-

"""
ATTENTION: NOTE: fitting a (larger) split grid model on the smaller subgrid
does NOT yield the same polynomial representation!
The two split parts are not orthogonal (not independent/separable) from each other
The Newton basis has this property:
the product "chains" in the divided differences get longer with increasing degree and are idependent
however in the Lagrange basis this does not hold:
the interpolation nodes seem to be equal to the smaller subgrid
and independent from each other in the larger split model
however, the Lagrange monomials! are not equal to the smaller model and they are dependent on each other
just because some coefficients will be 0, the representation is still different
only fitting on the full grid yields the same representation (the Lagrange coefficients won't be 0!)
TODO find proper transformation such the two bases are independent and separable
"""
import numpy as np

from minterpy.tree import MultiIndicesTree
from minterpy.utils import leja_ordered_values


def leja_split_values(n):
    # values of degree 2n (containing the values of degree n)
    # with an ordering such that the resulting interpolation grid will also contain the grid of degree n
    # TODO
    all_values = leja_ordered_values(n).squeeze()
    n_subgrid = n // 2
    subgrid_values = leja_ordered_values(n_subgrid).squeeze()
    values_out = np.empty(all_values.shape)
    # put the values of degree n first
    values_out[:n_subgrid + 1] = subgrid_values
    i = n_subgrid + 1
    for v in all_values:
        if v not in subgrid_values:
            values_out[i] = v
            i += 1
    return values_out.reshape((1, -1))


class SplitMultiIndicesTree(MultiIndicesTree):
    """
    class defining a Chebychev grid of degree 2n-1 ("split grid")
    such that the regular grid of degree n ("subgrid") is contained
    """

    def __init__(self, m: int, n: int, lp_degree: float = 2.0):
        # smallest possible subgrid is of degree 1 -> 2n = 2
        assert n >= 2, 'degree is too small. cannot build split grid.'
        assert n % 2 == 0, f'no split grid with degree {n} can be constructed. degree has to be even'
        super(SplitMultiIndicesTree, self).__init__(m, n, lp_degree)

    def _get_leja_vals(self):
        leja_values = leja_split_values(self.n)
        return leja_values


# TODO do not pass. assumptions do not hold! s. Note above. two split parts are not really separable
# the two fits are actually different
def test_split_grid_build(m, n):
    if n < 1:  # smallest possible subgrid degree is 1
        return

    print('\ntesting the split grid build...')
    # the method should suggest "missing interpolation node" as data samples
    r1 = MultivariatePolynomialRegression(m, n, verbose=True)
    n_split_grid = 2 * n
    r2 = MultivariatePolynomialRegression(m, n_split_grid, verbose=True, use_split_grid=True)

    t1 = r1.transformer.tree
    t2 = r2.transformer.tree
    v1 = t1.grid_values
    v2 = t2.grid_values
    nr_subgrid_values = v1.shape[1]
    np.testing.assert_equal(v1, v2[:, :nr_subgrid_values])

    # the indices of all grid points of the subgrid in the larger split grid
    matching_indices = find_matching_indices(t1.gamma, t2.gamma)
    np.testing.assert_equal(t1.gamma, t2.gamma[:, matching_indices])

    # test point construction:
    p1 = t1.grid_points
    p2 = t2.grid_points
    np.testing.assert_equal(p1, p2[:, matching_indices])

    coeffs_netwon_ground_truths = rnd_points(r1.N_fit)
    fct_ground_truth = lambda x: t1.eval(x, coeffs_netwon_ground_truths, verify_input=True)
    # fct_ground_truth = RUNGE_FCT_VECTORIZED
    interpol_nodes = p1
    fct_values = fct_ground_truth(interpol_nodes)
    r1.regression(fct_values, interpol_nodes)

    interpol_nodes2 = t2.grid_points
    fct_values2 = fct_ground_truth(interpol_nodes2)
    r2.regression(fct_values2, interpol_nodes2)

    # the transformation R of the split grid regressor is a "sparse" and shuffled identity matrix
    # (only some of its interpolation nodes are present)
    monomial_vals = r2.eval_lagrange_monomials_on(interpol_nodes)
    # np.testing.assert_allclose(monomial_vals, r2.R, rtol=NUMERICAL_TOLERANCE, atol=NUMERICAL_TOLERANCE)
    # assert np.sum(monomial_vals) == r1.N_fit

    c1 = r1.coeffs_lagrange
    c2 = r2.coeffs_lagrange
    np.testing.assert_allclose(c1, fct_values)
    # lagrange coefficients should match at the respective positions
    np.testing.assert_allclose(c1, c2[matching_indices])

    # coeffs_lagr_split = c2.copy()
    # # ... and be 0 at all other positions
    # coeffs_lagr_split[matching_indices] = 0.0
    # np.testing.assert_allclose(coeffs_lagr_split, 0.0)

    def equal_eval(pts):
        # evaluation should still yield the expected results:
        # using Newton eval
        fct_values = fct_ground_truth(pts)
        vals_newton1 = r1.evaluate_on(pts)
        vals_newton2 = r2.evaluate_on(pts)
        np.testing.assert_allclose(fct_values, vals_newton1)
        np.testing.assert_allclose(vals_newton1, vals_newton2)

        # ... and in Lagrange basis
        monomial_vals = r2.eval_lagrange_monomials_on(pts)
        coeffs_lagrange = r2.coeffs_lagrange
        vals_lagrange = monomial_vals @ coeffs_lagrange
        np.testing.assert_allclose(fct_values, vals_lagrange)

        coeffs_canonical = r2.transformer.transform_l2c(coeffs_lagrange)
        exponents = t2.gamma
        eval_canonical_fct = get_eval_fct_canonical(coeffs_canonical, exponents)
        val_canonical = np.apply_along_axis(eval_canonical_fct, 0, pts)
        np.testing.assert_allclose(val_canonical, vals_lagrange)

    def equal_grad(pts):
        # gradient should be remain equal:
        grad_eval1 = r1.eval_gradient_on(pts)
        grad_eval2 = r2.eval_gradient_on(pts)
        coeffs_newton = r2.coeffs_newton
        grad1 = r1.grad_newton
        grad2 = r2.grad_newton
        np.testing.assert_allclose(grad1, grad2)
        np.testing.assert_allclose(grad_eval1, grad_eval2)

        p0 = pts[:, 0]
        monomial_grad2 = r2.eval_lagrange_monomial_gradient_on(p0)
        monomial_grad2_p0 = monomial_grad2 @ r2.coeffs_lagrange
        grad2_p0 = grad_eval2[0, :]
        np.testing.assert_allclose(monomial_grad2_p0, grad2_p0)

    def test_eq_on(pts):
        equal_eval(pts)
        # equal_grad(pts)

    def test_equal_fit():
        test_eq_on(interpol_nodes)

        # also on any points (since fit is equal)
        nr_rnd_points = r1.N_fit
        scattered_points = rnd_points(m, nr_rnd_points)
        test_eq_on(scattered_points)

    test_equal_fit()

    # test how adding new points affects the transformation quality (e.g. if the split grid fit oscillates)
    new_point = rnd_points(m).reshape(m, 1)
    new_value = fct_ground_truth(new_point)

    r1.add_point(new_point, new_value)
    r2.add_point(new_point, new_value)

    new_point = new_point + (rnd_points(m).reshape(m, 1) / 1e3)
    new_value = fct_ground_truth(new_point)

    print(f'\n\nFIT QUALITY: (runge function dim {m})')
    print(f'\nregular deg {n} fit:')
    r1.add_point(new_point, new_value)
    interpolation_grid_eval = r1.sample_points
    eval_fit(interpolation_grid_eval, r1, fct_ground_truth)

    print(f'\ndeg ({n},{n_split_grid}) split fit:')
    r2.add_point(new_point, new_value)
    eval_fit(interpolation_grid_eval, r2, fct_ground_truth)
    # TODO compare fit quality. test
    # assert r1.error_measure > r2.error_measure

    r3 = MultivariatePolynomialRegression(m, n_split_grid, verbose=True, use_split_grid=False)
    t3 = r3.transformer.tree
    v3 = t3.grid_values
    # ordering is different -> grid structure is different
    if not np.array_equal(v2, v3):
        # also the transformation matrices should be different
        c2 = r2.transformer.lagrange2newton
        c3 = r3.transformer.lagrange2newton
        assert not np.allclose(c2, c3)
