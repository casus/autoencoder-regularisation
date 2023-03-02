# -*- coding:utf-8 -*-
import numpy as np

from minterpy.derivation import gradient_lagrange
from minterpy.transformation import Transformer
from minterpy.utils import rescale

"""
TODO use InterpolationTuner (Diploma Thesis Jannik Michelfeit)
 with a single (correct) model. with low numerical tolerance! -> optimised polynomial
TODO integrate directly into Interpolator class.
TODO add optimsiation ("hyper"-)parameters
TODO
IDEA: use the special properties of Lagrange polynomials to solve optimisation problem:
a coefficient of a Lagrange polynomial is the value of the polynomial at the corresponding interpolation node
-> by computing the gradient (in Lagrange form) the gradient on the interpolation grid is implicitly given
and can be used to optimise the function (polynomial)
global optimum of a function: extreme value + zero gradient
-> minimize norm of gradient while optimising function value
TODO for later:
"zoom in" -> evaluate the polynomial at a grid scaled around an area of interest
(defines a new lagrange polynomial) to get a more precise result.
recurse until convergence (threshold is reached)

 TODO what is more performant: gradient descent or resampling
  (evaluating the polynomial many times on a new grid, zooming in)?

ATTENTION:
    important to not "miss" global optimum and pick local optima instead!
    """
DEFAULT_ISCLOSE = np.isclose


# TODO histogram of gradients and values. print recursion. save
# TODO test
# TODO recursion
# TODO pass center point, TODO understand rescaling (interpolation always on unit hypercube)
# TODO use max distances between grid pts!
# TODO ATTENTION: all points must lie within the (interpolation) domain [-1;1]^m
def optimise(transformer: Transformer, coeffs_lagrange: np.ndarray, minimise=True,
             # val_magn_thres=1e-9,
             grad_magn_thres=1e-9, is_close_fct=DEFAULT_ISCLOSE, max_recursion_depth=3):
    fct_vals = coeffs_lagrange  # alias

    if minimise:
        pick_extreme = min
    else:
        pick_extreme = max

    def get_grid_pt(idx):
        return transformer.tree.grid_points[:, idx]

    def summary(idx):
        # point, value, gradient, gradient_mang
        return get_grid_pt(idx), fct_vals[idx], grad_lagrange[:, idx], grad_magnitudes[idx]

    extreme_val = pick_extreme(fct_vals)
    print(f'extreme value: {extreme_val}')
    val_is_extreme = is_close_fct(fct_vals, extreme_val)

    # TODO precompute and use gradient operator directly
    grad_lagrange = gradient_lagrange(transformer, coeffs_lagrange)
    grad_magnitudes = np.linalg.norm(grad_lagrange, axis=0)
    print(f'lowest gradient magnitude: {min(grad_magnitudes)}')
    grad_is_vanishing = grad_magnitudes < grad_magn_thres

    # NOTE: the case of optimal value and zero gradient should ALWAYS be favoured (= global optimum)
    is_optimal = val_is_extreme & grad_is_vanishing
    candidate_idxs = np.where(is_optimal)[0]
    if len(candidate_idxs) > 0:
        print(f'found candidates {candidate_idxs}')
        print(f'returning {candidate_idxs[0]}')
        return summary(candidate_idxs[0])

    max_recursion_depth -= 1
    if max_recursion_depth == 0:
        print('no optimum found with allowed recursion depth. terminating')
        return

    # no optimum has been found
    # increase degree -> grid gets more fine grained
    # NOTE: choose only even degrees (-> symmetric grids)
    construction_params = {'n': transformer.n + 2, 'm': transformer.m, 'lp_degree': transformer.lp_degree}
    transformer_new = Transformer(**construction_params)
    # just evaluate on new grid
    coeffs_lagrange_new = rescale(coeffs_lagrange, transformer, transformer_new)
    print(f'no candidate found. resampling with degree {construction_params["n"]} ({len(coeffs_lagrange_new)} points)')
    return optimise(transformer_new, coeffs_lagrange_new, max_recursion_depth=max_recursion_depth)

    # TODO there might be multiple candidates!
    # TODO refine until there is one unique candidate
    # = value more extreme and gradient smaller
    # TODO always possible? what about perfectly symmetric fcts?

    raise NotImplementedError('no optimum found:', extreme_val, fct_vals, grad_magnitudes)

    # observation: if the gradient at a point is large, the global optimum might be in vicinity,
    #  even if the function value is not optimal there!
    # TODO IDEA: "Bayesian" like approach: pick nodes to check based on a mix between function value (<- "performance")
    #  and gradient size (<- "uncertainty")
    # value <-> gradient
    # performance <-> uncertainty
    # exploration <-> exploitation
    # real Bayesian optimisation might be too expensive
    # identical behavior not required -> approximate
    # TODO softmax
