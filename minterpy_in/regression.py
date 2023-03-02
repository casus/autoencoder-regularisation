# -*- coding:utf-8 -*-

import time
from typing import Optional, Callable
from warnings import warn

import numpy as np
import scipy
from sklearn.linear_model import LinearRegression

from minterpy.derivation import get_gradient, get_gradient_operator
from minterpy.regression_helpers import is_stable_transform, invert, make_stable
from minterpy.transformation import Transformer
from minterpy.utils import report_error, rnd_points

# from skimage.util import view_as_blocks

INTERPOL_DOMAIN_ERROR_MSG = 'the grid points must fit the interpolation domain [-1;1]^m.'
CONDITION_NR_THRES = 3.0  # above which condition number a transformation is considered to be unstable


# TODO check input. all numpy arrays, no infty. nan etc.!
# TODO closely linked to Transformer ... classes. combine, inheritance?
# if cached transformation is being used, TODO assert that the sample points are actually equal
# TODO better: define API to regress on same samples only with fct values as input (use cached transform)
# TODO separate both. store sample points once during construction __init__(sample_points)
# TODO globally use transposed format of points. shape should be (np_points, m) to easily iterate over points.
#  more intuitive and practical
class MultivariatePolynomialRegression(object):

    def __init__(self, m: int, n: int, lp_degree: float = 2.0, verbose: bool = True,
                 ):
        self.m = m  # input dimension
        self.n = n  # model ("fit") degree
        self.lpDegree = lp_degree
        self.verbose = verbose
        if verbose:
            print(f'dim {m} deg {n}. building interpolation grid and conversion matrices...')
        self.transformer = Transformer(m=m, n=n, lp_degree=lp_degree)
        self.N_fit = self.transformer.N  # amount of coefficients and interpolation grid samples
        if verbose:
            print(f'... done. interpolation grid has {self.N_fit} points (= #coefficients)')
        self._coeffs_newton: Optional[np.ndarray] = None
        self._coeffs_lagrange: Optional[np.ndarray] = None
        self._sample_points: Optional[np.ndarray] = None
        self._transform_fit2data: Optional[np.ndarray] = None  # R
        self._transform_data2fit: Optional[np.ndarray] = None  # S
        self.grad_op_l2n: Optional[np.ndarray] = None
        self.grad_newton: Optional[np.ndarray] = None
        self.transformation_quality: float = 0.0

    # TODO these are actually attributes/functions a Polynomial/Solution class should have
    # just temporarily added them to Regressor class (highest level class)
    @property
    def transformation_stored(self):
        return self._sample_points is not None

    @property
    def coefficients_stored(self):
        return self._coeffs_newton is not None

    @property
    def gradient_stored(self):
        return self.grad_newton is not None

    @property
    def grad_op_stored(self):
        return self.grad_op_l2n is not None

    @property
    def R(self):
        assert self.transformation_stored
        return self._transform_fit2data

    @R.setter
    def R(self, R):
        N_data = R.shape[0]
        if self.verbose:
            print("Computing data to interpolation grid transformation.")
            print("data resolution %d; fit resolution %d" % (N_data, self.N_fit))
        start_time = time.time()
        # evaluate each Lagrange monomial on each sample point
        # this defines the transformation matrix from the interpolation grid to the data samples
        # TODO explain
        # "fit -> data"
        # shape: (N_data, N_fit)
        condition_nr = np.linalg.cond(R)

        if self.verbose:
            print(f'condition number of the transformation: {condition_nr:.2}')

        is_stable = is_stable_transform(R)
        if not is_stable:
            warn('the transformation is unstable')
            # raise np.linalg.LinAlgError('the transformation is unstable')

        # find S: the quasi "inverse" of R (= transformation from data to interpolation grid)
        # shape: (N_fit, N_data)
        S = invert(R, self.verbose)
        assert np.all(np.abs(S)) < 3.0
        # assert S.shape == (self.N_fit, N_data) # LU inverse is of shape N_min, N_min
        if self.verbose:
            print("computing the transformation took: %.2f s" % (time.time() - start_time))

        # now store the found (stable) transformation:
        self.transformation_quality = condition_nr
        self._transform_fit2data = R
        self._transform_data2fit = S
        # self.R_times_S = self.R @ self.S  # meaning: "autoencoder" like: compression + decompression

    @property
    def sample_points(self):
        assert self.transformation_stored
        return self._sample_points

    @sample_points.setter
    def sample_points(self, sample_points):
        if sample_points is None:  # reset the transformation
            self._transform_data2fit = None
            self._transform_fit2data = None
            self.transformation_quality = 0.0
            # TODO also reset the coefficients ("stored fit")

        m, N_data = sample_points.shape
        R = self.eval_lagrange_monomials_on(sample_points)  # might just have 1D
        R = R.reshape(N_data, self.N_fit)
        self.R = R  # computes S internally

        self._sample_points = sample_points  # defining the "data grid"

    def add_point(self, point, fct_value: Optional[float] = None, *args, **kwargs):
        # add entry to the respective matrices
        point = point.reshape(self.m, 1)
        sample_points = np.append(self.sample_points, point, axis=1)
        self.verify_sample_points(sample_points)
        # avoid completely recomputing R
        # NOTE: it is unlikely that a single added point will be exactly equal to an interpol. node
        # -> do not check for equality
        new_row = self.eval_lagrange_monomials_on(point).reshape(1, self.N_fit)
        self.R = np.append(self.R, new_row, axis=0)

        # NOTE: assign the private variable!
        self._sample_points = sample_points

        if fct_value is not None:
            self.add_fct_value(fct_value, *args, **kwargs)

    def add_fct_value(self, fct_value, regression_fct=None):
        # add entry to the respective matrices
        fct_values = np.append(self.fct_values, fct_value)
        self.verify_fct_vals(fct_values)
        assert len(fct_values) == self.N_data

        if regression_fct is None:
            # regression_fct = self.regression  # default
            regression_fct = self.simple_regression  # default

        return regression_fct(fct_values)

    @property
    def S(self):
        assert self.transformation_stored
        return self._transform_data2fit

    @property
    def N_data(self):
        return self.sample_points.shape[1]

    @property
    def coeffs_lagrange(self):
        assert self.coefficients_stored
        return self._coeffs_lagrange

    @coeffs_lagrange.setter
    def coeffs_lagrange(self, coeffs):
        self._coeffs_lagrange = coeffs
        if coeffs is None:
            self._coeffs_newton = None
        else:
            # automatically set the Newton coefficients once
            # ATTENTION: set private variable to avoid infinite recursion loop
            self._coeffs_newton = self.transformer.transform_l2n(coeffs)
        # gradient does not match the new coefficients -> reset
        self.grad_newton = None
        self.grad_op_l2n = None

    @property
    def coeffs_newton(self):
        assert self.coefficients_stored
        return self._coeffs_newton

    @coeffs_newton.setter
    def coeffs_newton(self, coeffs):
        self._coeffs_newton = coeffs
        if coeffs is None:
            self._coeffs_newton = None
        else:
            # automatically set the Lagrange coefficients once
            # ATTENTION: set private variable to avoid infinite recursion loop
            self._coeffs_lagrange = self.transformer.transform_n2l(coeffs)
        # gradient does not match the new coefficients -> reset
        self.grad_newton = None
        self.grad_op_l2n = None

    def evaluate_on(self, points):
        ''' evaluates the polynomial computed by a stored regression fit (given by self.coeffs_newton)
        on arbitrary points within the domain [-1,1]^m

        NOTE: supports 1D (single) and 2D points
        '''
        assert self.coefficients_stored
        # todo: replace eval_lp by Jannik's / Steve's evaluation function depending on backend (Python vs PyTorch)
        return self.transformer.tree.eval(points, self.coeffs_newton)

    def _precompute_gradient_op(self):
        """
        NOTE: with fixed interpolation nodes ("grid") the operation transforming coefficients into a gradient
        (= (m x N x N) tensor) can be precomputed
        NOTE: currently the Newton coefficients are used for polynomial evaluation (most efficient)
        -> store the gradient operator tensor in a format for easy evaluation -> to newton basis
        NOTE: for some applications we require to evaluate the gradient of the Lagrange monomials separately
            -> from Lagrange basis
        NOTE: TODO store in a format that causes the least numerical errors
        TODO literature: the "basic" transformations are: L2N (by interpolation), C2N
        NOTE: the gradient operator contains the Newton coefficients of every partial derivative Lagrange monomial
        NOTE: it is desirable to call the eval function just a single time,
            since all evaluations happen at the same point!
        NOTE: the columns of each 'lagrange2newton' gradient conversion matrix (<-> last axis 2 in the tensor)
          are the Newton coefficients of the respective (derivative) Lagrange monomials
        ATTENTION: axis 1 and 2 both have the proper coefficient length N
          -> make sure to iterate over the right axis
        """
        if self.grad_op_stored:
            return
        if self.verbose:
            print('computing gradient operator now...')
        transformer = self.transformer
        l2c = transformer.lagrange2canonical
        c2n = transformer.canonical2newton
        exponents = transformer.exponents
        # "gradient operator from Lagrange to Newton basis":
        # given the Lagrange coefficients outputs the Newton coefficients of all partial derivatives
        self.grad_op_l2n = get_gradient_operator(l2c, c2n, exponents)  # (m x N x N) tensor

        # NOTE: in order to easily evaluate these partial derivatives, store a "list of coefficients"
        # swap axes for a C-like ordering: (coefficient index < dimension < monomial)
        self.grad_lagr_mons = np.transpose(self.grad_op_l2n, (1, 0, 2))  # (N x m x N)
        # convert to 2D such that each column corresponds to a list of coefficients
        # as expected by the polynomial evaluation function
        N = self.N_fit
        self.grad_lagr_mons = self.grad_lagr_mons.reshape((N, -1), order='C')  # (N x mN)
        if self.verbose:
            print('... done.')

    def _precompute_gradient(self):
        if self.gradient_stored:
            return
        if not self.grad_op_stored:
            self._precompute_gradient_op()
        # with fixed coefficients the gradient can be precomputed
        # grad (N x m): Newton coefficients of the partial derivative in each dimension)
        # use format expected by polynomial evaluation function
        self.grad_newton = get_gradient(self.coeffs_lagrange, self.grad_op_l2n).T

    def eval_gradient_on(self, points):
        if not self.gradient_stored:
            self._precompute_gradient()
        coeffs_newton = self.grad_newton
        return self.transformer.tree.eval(points, coeffs_newton)

    def eval_lagrange_monomial_gradient_on(self, point):
        """ computes the values of the gradient of all Lagrange monomials at the given input point

        by default the evaluation of the gradient requires m * N polynomial evaluations
        NOTE: the gradient magnitudes of the Lagrange monomials remain high across the interpolation domain
        -> no computational tweaks to avoid the computation of some gradients
        based on distance of the evaluation points to the interpolation nodes possible

        Returns
        -------
        (m x N) the value of all partial derivative Lagrange monomials in each dimension
        """
        if not self.grad_op_stored:
            self._precompute_gradient_op()

        # the N Newton coefficients of all mN partial derivative Lagrange monomials
        gradient_coeffs = self.grad_lagr_mons  # in the expected input format (N x mN)
        grad_eval = self.transformer.tree.eval(point, gradient_coeffs)  # (mN)
        # C-like ordering -> earlier axis dimension m changes slower -> C-like ordering (m x N)
        grad_eval = np.reshape(grad_eval, (self.m, self.N_fit), order='C')
        return grad_eval

    def eval_lagrange_monomials_on(self, points):
        """ computes the values of all Lagrange monomials at all k input points

        NOTE: this is agnostic to the coefficient of the base polynomial
        NOTE: the columns of the 'lagrange2newton' conversion matrix are the Newton coefficients of the
          respective Lagrange monomials

        :param points: (m x k) the k points to evaluate on.
        :return: (k x N) the value of each Lagrange monomial in Newton form at each point.
        """
        coeffs_newton = self.transformer.lagrange2newton
        return self.transformer.tree.eval(points, coeffs_newton)

    def report_fit(self):
        print('error on the input samples:')
        report_error(self.fct_val_errors)

    def sample_points_are_valid(self, sample_points):
        # NOTE: avoid silently returning False when the input is not fulfilling any input constraints!
        self.verify_sample_points(sample_points)
        R = self.eval_lagrange_monomials_on(sample_points)
        is_stable = is_stable_transform(R)
        return is_stable

    def is_valid_new_point(self, point):
        # add entry to the respective matrices
        point = point.reshape(self.m, 1)
        sample_points = np.append(self.sample_points, point, axis=1)
        # NOTE: avoid silently returning False when the input point is not fulfilling any input constraints!
        self.verify_sample_points(sample_points)
        new_row = self.eval_lagrange_monomials_on(point)
        new_row = new_row.reshape(1, self.N_fit)
        R = np.append(self.R, new_row, axis=0)
        is_stable = is_stable_transform(R)
        return is_stable

    def propose_point(self):
        # IDEA: a transformation is numerically stable / accurate / expressive
        # if the data samples are similar to the interpolation grid
        # <-> the transformation matrix is similar to the identity matrix
        # question: which of the interpolation nodes has not been sampled?
        # answer: the one not having any "activation" close to 1 in the transformation matrix!
        # NOTE: might not be the sample maximising the matrix determinant
        # there must be more coefficients than data points -> still points (= interpol grid nodes) to suggest
        # TODO find approach for the case N_data >= N_fit:
        #  use linear algebra, maximise matrix determinant, suggest matrix entry, generate sample
        #  -> towards identity transformation matrix
        assert self.transformation_stored, 'cannot propose points without having a transformation stored'
        assert self.sample_points.shape[1] < self.N_fit
        max_activations = np.max(np.abs(self.R), axis=0)  # ATTENTION: entries in R can also be negative!
        assert len(max_activations) == self.N_fit
        node_idxs = np.argsort(max_activations)
        for node_idx in node_idxs:
            point = self.transformer.tree.grid_points[:, node_idx]
            # test if transformation would work with this point
            if self.is_valid_new_point(point):
                return point
        # none of the default grid points is a valid sample. try with random points
        for _ in range(100):
            point = rnd_points(self.m)
            if self.is_valid_new_point(point):
                return point

        raise ValueError('no valid point could be suggested. all points seem to cause instabilities')

        # TODO more sophisticated search

    def propose_n_add_point(self):
        # TODO reuse already internally computed transformation!
        point = self.propose_point()
        self.add_point(point)
        return point

    def make_stable(self, condition_nr_thres=CONDITION_NR_THRES, fct_values=None):
        if fct_values is None:
            fct_values = self.fct_values
        R, points, fct_values = self.R, self.sample_points, self.fct_values
        R, points, fct_values, condition_nr, removed_pts_idxs = make_stable(R, points, fct_values, condition_nr_thres)
        nr_removed_pts = len(removed_pts_idxs)
        if nr_removed_pts > 0:
            self.R = R
            self._sample_points = points  # update private feature
            self.coeffs_newton = None

        return removed_pts_idxs

    # general version: able to handle both under- and overfitting
    def cache_transform(self, sample_points, verify_input: bool = False):
        if verify_input:
            self.verify_sample_points(sample_points)
        self.sample_points = sample_points

    def _return_signature(self):  # unifies the return parameters of all regression fcts
        # TODO decide
        return self.fct_values_regr, self.error_measure, self.coeffs_lagrange

    def verify_fct_vals(self, fct_values):
        assert not np.any(np.isnan(fct_values))
        assert not np.any(np.isinf(fct_values))
        # TODO test numerical properties to assert numerical stability of regression

    def equal_sample_points_stored(self, sample_points):
        if not self.transformation_stored:
            return False
        if self.sample_points.shape != sample_points.shape:
            return False
        return np.allclose(self.sample_points, sample_points)

    def verify_sample_points(self, sample_points):
        m, nr_data_points = sample_points.shape  # ensure that dimensions fit
        assert m == self.m
        self.verify_fct_vals(sample_points)
        sample_max = np.max(sample_points, axis=1)
        assert np.allclose(np.maximum(sample_max, 1.0), 1.0), \
            INTERPOL_DOMAIN_ERROR_MSG + f'violated max: {sample_max}'
        sample_min = np.min(sample_points, axis=1)
        assert np.allclose(np.minimum(sample_min, -1.0), -1.0), \
            INTERPOL_DOMAIN_ERROR_MSG + f'violated min: {sample_min}'
        max_grid_val = np.max(sample_points)
        if not np.isclose(max_grid_val, 1.0):
            warn(f'the highest encountered value in the grid is {max_grid_val}  (expected 1.0). '
                 'this can cause rank deficiencies in the conversion matrices. '
                 'ensure to use the appropriate conversion matrices in case of a scaled input grid!',
                 category=UserWarning)
        min_grid_val = np.min(sample_points)
        if not np.isclose(min_grid_val, -1.0):
            warn(f'the smallest encountered value in the grid is {min_grid_val} (expected -1.0). '
                 'this can cause rank deficiencies in the conversion matrices. '
                 'ensure to use the appropriate conversion matrices in case of a scaled input grid!',
                 category=UserWarning)

        # TODO sample points must be unique (no point pair too similar)
        # TODO test numerical properties -> stability

    def verify_input(self, fct_values, sample_points):
        # TODO check correct dtypes

        nr_fct_vals = len(fct_values)
        self.verify_fct_vals(fct_values)

        if sample_points is not None:
            m, nr_data_points = sample_points.shape  # ensure that dimensions fit
            assert nr_fct_vals == nr_data_points, 'for every sample point a function value has to be given'
            self.verify_sample_points(sample_points)
            # check if the input has changed -> re-computation of the transformation required
            self.transformation_cached = self.equal_sample_points_stored(sample_points)
        elif self.transformation_stored:
            assert nr_fct_vals == self.N_data, 'for every sample point a function value has to be given'

    # TODO use. test
    def regress_runge_regularized(self, fct_values, verify_input: bool = False, lambda_: float = 0.001):
        if self.verbose:
            print("ridge regularised regression")
        if verify_input:
            self.verify_fct_vals(fct_values)
        # define aliases
        S = self.S
        R = self.R
        f = fct_values
        eta_rungePenalty_ls = np.linalg.inv(S.T @ R.T @ R @ S + lambda_ * S.T @ S) @ S.T @ R.T @ f
        coeffs_lagrange = S @ eta_rungePenalty_ls
        assert len(coeffs_lagrange) == self.N_fit, \
            f'this regression does not work when N_data ({self.N_data} < N_fit ({self.N_fit})'
        return coeffs_lagrange

    def regress_refined(self, fct_values, verify_input: bool = False):
        if self.verbose:
            print("refined regression")
        if verify_input:
            self.verify_fct_vals(fct_values)
        # aliases:
        f = fct_values
        S = self.S
        R = self.R
        # "least square" fit:
        eta = np.linalg.inv(S.T @ R.T @ R @ S) @ S.T @ R.T @ f
        coeffs_lagrange = self.S @ eta
        assert len(coeffs_lagrange) == self.N_fit, \
            f'this regression does not work when N_data ({self.N_data} < N_fit ({self.N_fit})'
        return coeffs_lagrange

    def regress_simple(self, fct_values, verify_input: bool = False):
        if self.verbose:
            print("simple polynomial regression")
        if verify_input:
            self.verify_fct_vals(fct_values)
        coeffs_lagrange, _, _, _ = scipy.linalg.lstsq(self.R, fct_values)
        assert len(coeffs_lagrange) == self.N_fit, \
            f'wrong coefficient amount: ' \
            f'#coefficients: {len(coeffs_lagrange)} != N_fit {self.N_fit},  N_data: {self.N_data}'
        return coeffs_lagrange

    def regress_weighted(self, fct_values, sample_weights, verify_input: bool = False):
        if self.verbose:
            print("weighted polynomial regression")
        if verify_input:
            self.verify_fct_vals(fct_values)
            assert fct_values.shape == sample_weights.shape
            assert np.all(sample_weights >= 0.0)
        X = self.R
        y = fct_values
        reg = LinearRegression().fit(X, y, sample_weight=sample_weights)
        coeffs_lagrange = reg.coef_
        assert len(coeffs_lagrange) == self.N_fit, \
            f'wrong coefficient amount: ' \
            f'#coefficients: {len(coeffs_lagrange)} != N_fit {self.N_fit},  N_data: {self.N_data}'
        return coeffs_lagrange

    def compute_error_measure(self, vals_true):
        assert vals_true.shape == self.fct_values_regr.shape
        self.error_measure = np.mean(np.abs(self.fct_val_errors))

    def _regr_wrapper(self, core_regression_fct: Callable,
                      fct_values: np.ndarray, sample_points: Optional[np.ndarray] = None,
                      use_cached_transform: bool = False, verify_input: bool = True,
                      *args, **kwargs):
        """ defines equal behaviour of all regression fcts      """
        if verify_input:
            self.verify_input(fct_values, sample_points)

        if self.transformation_stored and use_cached_transform or sample_points is None:
            if self.verbose:
                print('using cached transform.')
            msg = 'attempting to use cached transform, but no transformation has been stored yet.'
            assert self.transformation_stored, msg
        else:
            self.cache_transform(sample_points)  # independent of the function values
        self.fct_values = fct_values
        start_time = time.time()
        # independent on the sample points (transformation computed earlier)
        # the Lagrange coeffs (=values) of the polynomial on the interpolation nodes (grid)
        self.coeffs_lagrange = core_regression_fct(fct_values, verify_input=verify_input, *args, **kwargs)
        # Lagrange coefficients (= values) of the polynomial on the interpolation grid
        assert len(self.coeffs_lagrange) == self.N_fit

        # the Lagrange coeffs (=values) of the polynomial on the data (/sample) grid
        # previously:
        # H = P1 @ f
        # H = H[0:N] or
        # H = H[0:N_min]
        # other formulation:
        # self.fct_values_regr = self.R_times_S @ eta
        # NOTE: also able to compute this with self.evaluate_on(sample_points), but slower!
        self.fct_values_regr = self.R @ self.coeffs_lagrange  # transform from interpol to data grid
        assert len(self.fct_values_regr) == len(fct_values)
        self.fct_val_errors = self.fct_values_regr - fct_values

        if self.verbose:
            fit_time = time.time() - start_time
            print(f'fit took {fit_time:.2f}s')
            self.report_fit()

        self.compute_error_measure(fct_values)
        return self._return_signature()

    def regression(self, fct_values: np.ndarray, sample_points: Optional[np.ndarray] = None,
                   use_cached_transform: bool = False, verify_input: bool = True):
        """fits a polynomial using refined least-squares regression approach

        make sure that the points `grid_points` are normalized within range [-1,1]
        """
        return self._regr_wrapper(self.regress_refined, fct_values, sample_points, use_cached_transform, verify_input)

    def simple_regression(self, fct_values: np.ndarray, sample_points: Optional[np.ndarray] = None,
                          use_cached_transform: bool = False, verify_input: bool = True):
        ''' fits polynomial using simple regression approach.

        make sure that the points `grid_points` are normalized within range [-1,1]
        '''
        return self._regr_wrapper(self.regress_simple, fct_values, sample_points, use_cached_transform, verify_input)

    def weighted_regression(self, fct_values: np.ndarray, sample_weights: np.ndarray,
                            sample_points: Optional[np.ndarray] = None,
                            use_cached_transform: bool = False, verify_input: bool = True):
        ''' fits polynomial using a weighted regression approach.

        make sure that the points `grid_points` are normalized within range [-1,1]
        '''
        return self._regr_wrapper(self.regress_weighted, fct_values, sample_points, use_cached_transform, verify_input,
                                  sample_weights=sample_weights)

    # FIXME lambda unused
    def windowed_regression(self, image, block_shape=(64, 64), useRefinedRegression=True, lambda_=0.001):

        '''
        Fit polynomial on non-overlapping subpatches of our image specified by
        @block_shape
        '''
        # size of blocks
        image_shape = image.shape

        # define coordinates
        x = np.arange(block_shape[0])
        y = np.arange(block_shape[1])
        x0, y0 = np.meshgrid(x, y)
        x0 = (x0.reshape((-1, 1)))
        y0 = (y0.reshape((-1, 1)))
        coord = np.concatenate([x0, y0], 1)
        centerCoordinate = (block_shape[0] - 1) / 2
        coord = ((coord - centerCoordinate) / centerCoordinate).transpose()
        view = view_as_blocks(image, block_shape).copy()

        F = view[0, 0, :, :].reshape(-1)

        if useRefinedRegression:
            self.cache_transform(F, coord)

        no_blocks = view.shape[0] * view.shape[1]
        print("No blocks: %.4f" % (no_blocks))
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):
                start_time = time.time()
                F = view[i, j, :, :].squeeze().reshape(-1)
                if useRefinedRegression:
                    F_hat, err, H = self.regression(F, coord, use_cached_transform=True)
                else:
                    F_hat, err, H = self.simple_regression(F, coord)
                if self.verbose:
                    print("[%d,%d] err %.2f %.2fs" % (i, j, err, time.time() - start_time))
                view[i, j, :, :] = F_hat.reshape(block_shape)

        blocked_reshaped = view.transpose(0, 2, 1, 3).reshape((512, 512))
        return blocked_reshaped
