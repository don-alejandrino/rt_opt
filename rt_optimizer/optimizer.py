import itertools
import warnings

import numpy as np
from scipy import spatial, special
from scipy.optimize import OptimizeResult

from rt_optimizer.global_search import run_and_tumble
from rt_optimizer.local_search import bfgs_b


def _prepare_bounds(bounds, n_dims):
    """
    Check size and validity of a rectangular bounds object, and turn it into the required format for
    the following calculations.

    :param bounds: [array-like object] Rectangular bounds object
    :param n_dims: [int] Dimensionality of the problem
    :return: (bound_lower [np.array], bound_upper [np.array])
    """

    if bounds is not None:
        bounds = np.array(bounds)
        if n_dims is not None:
            assert bounds.shape == (n_dims, 2), ('bounds has wrong shape. Expected shape: ' +
                                                 '(n_dims, 2), where n_dims is the ' +
                                                 'dimensionality of the problem.')
        bound_lower = bounds[:, 0]
        bound_upper = bounds[:, 1]
        assert (bound_upper > bound_lower).all(), ('Upper bound must always be larger than lower ' +
                                                   'bound.')

        return bound_lower, bound_upper

    else:
        assert n_dims is not None, 'If bounds is None, n_dims must be provided.'

        return np.repeat(-np.inf, n_dims), np.repeat(np.inf, n_dims)


def _prepare_x0(x0, n_bacteria_per_dim, n_reduced_dims):
    """
    Check and prepare initial conditions object x0. If x0 is a vector, i.e., if it has the shape
    (n_dims,) it is duplicated times the total number of bacteria, which is given by
    n_bacteria_per_dim ** min(n_dims, n_reduced_dims).

    :param x0: [array-like object] Initial conditions object. Must have the shape
           (n_bacteria, n_dims) or (n_dims,)
    :param n_bacteria_per_dim: [int] Number of bacteria for each dimension
    :param n_reduced_dims: [int] Number of reduced dimensions used by the sequential random
           embeddings algorithm
    :return: Initial conditions for all bacteria [np.array of shape (n_bacteria, n_dims)]
    """

    x0 = np.array(x0)
    if len(x0.shape) == 1:
        n_dims = x0.shape[0]
        n_bacteria = n_bacteria_per_dim ** min(n_dims, n_reduced_dims)
        x0_population = np.tile(x0, (n_bacteria, 1))
    elif len(x0.shape) == 2:
        n_dims = x0.shape[1]
        n_bacteria = x0.shape[0]
        if n_bacteria != n_bacteria_per_dim ** min(n_dims, n_reduced_dims):
            warnings.warn('The number of bacteria given by x0 does not match the number of ' +
                          'bacteria given by the relation n_bacteria = ' +
                          'n_bacteria_per_dim ** min(n_dims, n_reduced_dims). The latter implies ' +
                          f'that n_bacteria = {n_bacteria_per_dim ** min(n_dims, n_reduced_dims)}, ' +
                          f'whereas the former implies that n_bacteria = {n_bacteria}. Using ' +
                          f'n_bacteria = {n_bacteria}.')
        x0_population = x0.copy()
    else:
        raise ValueError('x0 must be an array of either the shape (n_bacteria, n_dims) or ' +
                         '(n_dims,).')

    return x0_population


def _pad_trace(trace, targetLength):
    """
    Pad single-bacteria trace to given length.

    :param trace: [np.array] Single-bacteria trace
    :param targetLength: [int] Desired length
    :return: Padded trace [np.array]
    """

    currentLength = trace.shape[0]
    paddingLength = (targetLength - currentLength)

    return np.pad(trace, [(0, paddingLength), (0, 0)], mode="edge")


def _sequential_random_embeddings(f, x0, bounds, orig_dim, n_reduced_dims=3, n_embeddings=10,
                                  verbosity=1, **optimizer_kwargs):
    """
    Implementation the Sequential Random Embeddings algorithm described in
    +++++
    H. Qian, Y.-Q. Hu, and Y. Yu, Derivative-Free Optimization of High-Dimensional Non-Convex
    Functions by Sequential Random Embeddings, Proceedings of the Twenty-Fifth International Joint
    Conference on Artificial Intelligence, AAAI Press (2016).
    +++++
    The idea is basically to reduce high-dimensional problems to low-dimensional ones by embedding
    the original, high-dimensional search space ℝ^h into a low dimensional one, ℝ^l, by
    sequentially applying the random linear transformation
    x(n+1) = α(n+1)x(n) + A•y(n+1),    x ∈ ℝ^h, y ∈ ℝ^l, A ∈ N(0, 1)^(h×l), α ∈ ℝ
    and minimizing the objective function f(αx + A•y) w.r.t. [a, y].

    :param f: [callable] Objective function
    :param bounds: [callable] Bounds projection. The function bounds(x) must return a tuple
           (x_projected, bounds_hit), where x_projected is the input variable x projected to the
           defined the defined search region. That is, if x is within this region, it is returned
           unchanged, whereas if it is outside this region, it is projected to the region's
           boundaries. The second output, bounds_hit, indicates whether the boundary has been hit
           for each component of x. If, for example, x is three-dimensional and has hit the search
           region's boundaries in x_1 and x_2, but not in x_3, bounds_hit = [True, True, False].
           Note that the search domain needs not necessarily be rectangular. Therefore, we define a
           "boundary hit" in any component of x in the following way:
           bounds_hit[i] = True iff either x + δê_i or x - δê_i is outside the defined search
           domain ∀ δ ∈ ℝ⁺, where ê_i is the i_th unit vector
    :param orig_dim: [int] Original dimension of the problem, ℝ^h
    :param n_reduced_dims: [int] Dimension of the embedded problem, ℝ^(l+1)
    :param n_embeddings: [int] Number of embedding iterations
    :param verbosity: [int] Output verbosity. Must be 0, 1, or 2
    :param optimizer_args: [dict] Arguments to pass to the actual optimization routine
    :return: Best minimum of f found [scipy.optimize.OptimizeResult]
    """

    assert verbosity in [0, 1, 2], 'verbosity must be 0, 1, or 2.'

    x = np.zeros(orig_dim)
    x_best = x.copy()
    f_best = np.inf
    nfev = nit = 0
    success_best = False
    for i in range(n_embeddings):
        A = np.random.normal(size=(orig_dim, n_reduced_dims - 1))

        # Normalize rows of A
        normalization_sum = A.sum(axis=1)
        normalization_sum = np.where(normalization_sum == 0, 1, normalization_sum)
        A = A / normalization_sum[:, np.newaxis]

        def f_embedded(arg): return f(bounds(arg[0] * x + A.dot(arg[1:]))[0])

        # Set up bounds callback
        def bounds_embedded(arg):
            bounds_hit = np.zeros(len(arg), dtype=bool)
            x_proj, bounds_hit_orig = bounds(arg[0] * x + A.dot(arg[1:]))
            if bounds_hit_orig.any():  # Boundary hit in original, non-embedded variable
                arg[1:] = np.linalg.lstsq(A, x_proj - arg[0] * x, rcond=None)[0]
                bounds_hit[1:] = (A[bounds_hit_orig] != 0).any(axis=0)

            return arg, bounds_hit

        # Set up y0
        y0 = np.zeros((x0.shape[0], n_reduced_dims))
        y0[:, 0] = 1
        y0[:, 1:] = np.array([np.linalg.lstsq(A, x_orig - x, rcond=None)[0] for x_orig in x0])

        if verbosity > 0:
            infoMsg = f'\nEmbedding iteration {i}'
            print(infoMsg)
            print('-' * len(infoMsg))

        optimizer_kwargs['verbosity'] = verbosity
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='Found initial conditions outside the defined search domain.'
            )
            res_embedded = optimize(f_embedded, x0=y0, bounds=bounds_embedded, **optimizer_kwargs)
        y = res_embedded.x
        f_val = res_embedded.fun
        nfev += res_embedded.nfev
        nit += res_embedded.nit

        x = bounds(y[0] * x + A.dot(y[1:]))[0]

        if verbosity > 0:
            print(f'Random embedding gave x = {x}.')

        if f_val < f_best:
            f_best = f_val
            x_best = x.copy()
            success_best = res_embedded.success

    result = OptimizeResult()
    result.success = success_best
    result.x = x_best
    result.fun = f_best
    result.nfev = nfev
    result.nit = nit
    result.trace = None

    return result


def optimize(f, x0=None, bounds=None, domain_scale=None, init='uniform', stepsize_start=None,
             stepsize_decay_fac=1e-3, base_tumble_rate=0.1, niter_rt=400, n_bacteria_per_dim=3,
             stationarity_window=20, eps_stat=1e-3, attraction=False, attraction_window=10,
             attraction_sigma=None, attraction_strength=0.5, bounds_reflection=False,
             n_best_selection=3, c_gd=1e-6, a_gd=None, n_linesearch_gd=20, alpha_linesearch_gd=0.5,
             beta_linesearch_gd=0.33, eps_abs_gd=1e-9, eps_rel_gd=1e-6, niter_gd=100,
             n_embeddings=5, max_dims=3, n_reduced_dims=2, verbosity=0):
    """
    ToDo: Write docstring

    :param f:
    :param x0:
    :param bounds:
    :param domain_scale:
    :param init:
    :param stepsize_start:
    :param stepsize_decay_fac:
    :param base_tumble_rate:
    :param niter_rt:
    :param n_bacteria_per_dim:
    :param stationarity_window:
    :param eps_stat:
    :param attraction:
    :param attraction_window:
    :param attraction_sigma:
    :param attraction_strength:
    :param bounds_reflection:
    :param n_best_selection:
    :param c_gd:
    :param a_gd:
    :param n_linesearch_gd:
    :param alpha_linesearch_gd:
    :param beta_linesearch_gd:
    :param eps_abs_gd:
    :param eps_rel_gd:
    :param niter_gd:
    :param n_embeddings:
    :param max_dims:
    :param n_reduced_dims:
    :param verbosity:
    :return:
    """

    assert verbosity in [0, 1, 2], 'verbosity must be 0, 1, or 2.'
    assert n_reduced_dims >= 2, 'n_reduced_dims must not be less than 2.'

    n_reduced_dims_eff = n_reduced_dims + 1

    if bounds is None or callable(bounds):
        assert x0 is not None, ('If no box constraints are provided for bounds, x0 must not be ' +
                                'None.')
        x0_population = _prepare_x0(x0, n_bacteria_per_dim, n_reduced_dims_eff)
        n_bacteria, n_dims = x0_population.shape

        if bounds is None:
            bound_lower, bound_upper = _prepare_bounds(bounds, n_dims)

            def projection_callback(x):
                x = np.clip(x, bound_lower, bound_upper)
                bounds_hit = np.where(((x == bound_lower) | (x == bound_upper)), True, False)
                return x, bounds_hit

            def projection_callback_population(x):
                return projection_callback(x)

        else:

            def projection_callback(x):
                return bounds(x)

            def projection_callback_population(x):
                out = np.array([projection_callback(x_single) for x_single in x])
                return out[:, 0], out[:, 1]

    elif isinstance(bounds, (list, np.ndarray)):
        if x0 is not None:
            x0_population = _prepare_x0(x0, n_bacteria_per_dim, n_reduced_dims_eff)
            n_bacteria, n_dims = x0_population.shape
            bound_lower, bound_upper = _prepare_bounds(bounds, n_dims)
        else:
            bound_lower, bound_upper = _prepare_bounds(bounds, None)
            n_dims = len(bound_lower)
            n_bacteria = n_bacteria_per_dim ** min(n_dims, n_reduced_dims_eff)
            if init == 'uniform' and n_dims > n_reduced_dims_eff:
                init = 'random'
                if verbosity > 0:
                    warnings.warn('The option init="uniform" is only available for problems with ' +
                                  f'dimensionality less than or equal to {n_reduced_dims_eff}. ' +
                                  f'Since the current problem has dimensionality {n_dims}, init '
                                  'was automatically set to "random".')
            if init == 'random':
                x0_population = np.random.uniform(bound_lower, bound_upper,
                                                  size=(n_bacteria, n_dims))
            elif init == 'uniform':
                init_points = []
                for i in range(n_dims):
                    init_points.append(np.linspace(bound_lower[i], bound_upper[i],
                                                   n_bacteria_per_dim))
                x0_population = np.array(np.meshgrid(*init_points)).reshape(n_dims, -1).T
            else:
                raise ValueError('init must either be "random" or "uniform".')

        def projection_callback(x):
            x = np.clip(x, bound_lower, bound_upper)
            bounds_hit = np.where(((x == bound_lower) | (x == bound_upper)), True, False)
            return x, bounds_hit

        def projection_callback_population(x):
            return projection_callback(x)

    else:
        raise ValueError('bounds must either be None, an array or corresponding nested list of ' +
                         'shape (n_dims, 2), or a custom callback function. See the docstring ' +
                         'for details.')

    assert niter_rt > stationarity_window, 'niter_rt must be larger than stationarity_window.'
    assert n_best_selection <= n_bacteria, 'n_best_selection must not be larger than n_bacteria.'

    if stepsize_start is not None:
        auto_scale_stepsize = False
    else:
        auto_scale_stepsize = True
        stepsize_start = 1e-1
    stepsize_end = stepsize_decay_fac * stepsize_start

    if attraction_sigma is not None:
        auto_scale_attraction_sigma = False
    else:
        auto_scale_attraction_sigma = True
        attraction_sigma = 1

    if a_gd is not None:
        auto_scale_a_gd = False
    else:
        auto_scale_a_gd = True
        a_gd = 1e-2

    x0_population_orig = x0_population.copy()
    x0_population, _ = projection_callback_population(x0_population)
    if not np.array_equal(x0_population, x0_population_orig):
        warnings.warn('Found initial conditions outside the defined search domain.')

    max_scale = None
    if domain_scale is not None:
        max_scale = domain_scale
    elif isinstance(bounds, (list, np.ndarray)):
        # noinspection PyUnboundLocalVariable
        domain_range = bound_upper - bound_lower
        max_scale = np.max(np.where(np.isinf(domain_range), 0, domain_range))
    if max_scale is not None and max_scale > 0:
        if auto_scale_stepsize:
            stepsize_start = stepsize_start * max_scale
            stepsize_end = stepsize_end * max_scale
        if auto_scale_attraction_sigma:
            attraction_sigma = attraction_sigma * max_scale
        if auto_scale_a_gd:
            a_gd = a_gd * max_scale

    if n_dims > max_dims:
        if verbosity > 0:
            print(f'Using sequential random embeddings in {n_reduced_dims} + 1 dimensions.')
        return _sequential_random_embeddings(f,
                                             x0_population,
                                             projection_callback,
                                             n_dims,
                                             n_reduced_dims=n_reduced_dims_eff,
                                             n_embeddings=n_embeddings,
                                             verbosity=verbosity,
                                             domain_scale=max_scale,
                                             init=init,
                                             stepsize_start=stepsize_start,
                                             stepsize_decay_fac=stepsize_decay_fac,
                                             base_tumble_rate=base_tumble_rate,
                                             niter_rt=niter_rt,
                                             n_bacteria_per_dim=n_bacteria_per_dim,
                                             stationarity_window=stationarity_window,
                                             eps_stat=eps_stat,
                                             attraction=attraction,
                                             attraction_window=attraction_window,
                                             attraction_sigma=attraction_sigma,
                                             attraction_strength=attraction_strength,
                                             bounds_reflection=bounds_reflection,
                                             n_best_selection=n_best_selection,
                                             c_gd=c_gd,
                                             a_gd=a_gd,
                                             n_linesearch_gd=n_linesearch_gd,
                                             alpha_linesearch_gd=alpha_linesearch_gd,
                                             beta_linesearch_gd=beta_linesearch_gd,
                                             eps_abs_gd=eps_abs_gd,
                                             eps_rel_gd=eps_rel_gd,
                                             niter_gd=niter_gd)

    else:
        x_best, f_best, nfev, nit, trace = run_and_tumble(f,
                                                          x0_population,
                                                          projection_callback_population,
                                                          niter_rt,
                                                          stepsize_start,
                                                          stepsize_end,
                                                          base_tumble_rate=base_tumble_rate,
                                                          stationarity_window=stationarity_window,
                                                          eps_stat=eps_stat,
                                                          attraction=attraction,
                                                          attraction_window=attraction_window,
                                                          attraction_sigma=attraction_sigma,
                                                          attraction_strength=attraction_strength,
                                                          bounds_reflection=bounds_reflection,
                                                          verbosity=verbosity)

        if verbosity == 2:
            print('===============================================================================')
        if verbosity > 0:
            print(f'Best result after run-and-tumble stage is x = {x_best[np.argmin(f_best)]}, ' +
                  f'f(x) = {np.min(f_best)}. Starting local, gradient-based optimization for the ' +
                  f'{n_best_selection} best bacteria.')

        sortIdx = f_best.argsort()
        x_best_selection = x_best[sortIdx[:n_best_selection]]
        x_best_gd = np.empty(x_best_selection.shape)
        f_min_gd = np.empty(n_best_selection)
        nfev_gd = 0
        nit_gd = 0
        success_gd = np.empty(n_best_selection)
        trace_gd = np.empty((niter_gd, n_bacteria, n_dims))
        trace_gd[:, sortIdx[n_best_selection:], :] = trace[-1, sortIdx[n_best_selection:], :]
        nit_gd_arr = np.empty(n_best_selection)
        visited_points = trace.reshape(-1, n_dims)

        for n, x_start in enumerate(x_best_selection):
            if verbosity == 2:
                print(f'Performing gradient descent for bacterium {n}.')

            # Calculate quadratic function approximation around x_start
            num_sampling_points = 2 * int(special.binom(n_dims + 2, 2))
            # noinspection PyArgumentList,PyUnresolvedReferences
            sampling_points = visited_points[
                spatial.cKDTree(visited_points).query(x_start, num_sampling_points)[1]
            ]
            func_values = np.array([f(point) for point in sampling_points])
            nfev += num_sampling_points
            polynomial_powers = list(itertools.filterfalse(lambda prod: sum(list(prod)) > 2,
                                                           itertools.product((0, 1, 2),
                                                                             repeat=n_dims)))
            sampling_matrix = np.stack([np.prod(sampling_points ** d, axis=1)
                                        for d in polynomial_powers], axis=-1)
            coeffs = np.linalg.lstsq(sampling_matrix, func_values, 2)[0]

            # Calculate Hessian matrix from the quadratic approximation
            H = np.ones((n_dims, n_dims))
            square_powers = list(itertools.filterfalse(
                lambda zipped_item: sum(list(zipped_item[0])) != 2, zip(polynomial_powers, coeffs))
            )
            for square_power, coeff in square_powers:
                idcs_to_consider = np.argwhere(np.array(square_power) != 0)
                if len(idcs_to_consider) == 1:  # Diagonal
                    H[idcs_to_consider[0], idcs_to_consider[0]] = 0.5 * coeff
                elif len(idcs_to_consider) == 2:  # Mixed derivatives
                    H[idcs_to_consider[0], idcs_to_consider[1]] = coeff
                    H[idcs_to_consider[1], idcs_to_consider[0]] = coeff
                else:
                    raise RuntimeError("Polynomial function approximation seems to be of higher "
                                       "order than two. This shouldn't happen.")

            (x_gd_single,
             f_min_gd_single,
             nfev_gd_single,
             nit_gd_single,
             success_gd_single,
             trace_gd_single) = bfgs_b(f,
                                       x_start,
                                       projection_callback,
                                       H_start=H,
                                       a=a_gd,
                                       c=c_gd,
                                       niter=niter_gd,
                                       n_linesearch=n_linesearch_gd,
                                       alpha_linesearch=alpha_linesearch_gd,
                                       beta_linesearch=beta_linesearch_gd,
                                       eps_abs=eps_abs_gd,
                                       eps_rel=eps_rel_gd,
                                       verbosity=verbosity)
            x_best_gd[n] = x_gd_single
            f_min_gd[n] = f_min_gd_single
            nfev_gd += nfev_gd_single
            nit_gd += nit_gd_single
            nit_gd_arr[n] = nit_gd_single
            success_gd[n] = success_gd_single
            trace_gd[:, sortIdx[n], :] = _pad_trace(trace_gd_single, niter_gd)

        result = OptimizeResult()
        result.success = success_gd.any()
        result.x = x_best_gd[np.argmin(f_min_gd)]
        result.fun = np.min(f_min_gd)
        result.nfev = nfev + nfev_gd
        result.nit = nit + nit_gd
        trace_gd = trace_gd[:np.max(nit_gd_arr).astype(int)]
        result.trace = np.concatenate((trace, trace_gd))

        return result
