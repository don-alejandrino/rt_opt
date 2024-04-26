import itertools
import warnings

import numpy as np
from scipy import spatial, special
from scipy.optimize import OptimizeResult

from rt_opt.global_search import run_and_tumble
from rt_opt.local_search import bfgs_b


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


def _prepare_x0(x0, n_bacteria_per_dim, max_dims, n_reduced_dims_eff):
    """
    Check and prepare initial conditions object x0. If x0 is a vector, that is, if it has the shape
    (n_dims,) it is duplicated times the total number of bacteria, which is given by
    i)  n_bacteria = n_bacteria_per_dim ** n_dims if n_dims <= max_dims or
    ii) n_bacteria = n_bacteria_per_dim ** n_reduced_dims_eff if n_dims > max_dims.

    :param x0: [array-like object] Initial conditions object. Must have the shape
           (n_bacteria, n_dims) or (n_dims,)
    :param n_bacteria_per_dim: [int] Number of bacteria for each dimension
    :param max_dims: [int] Maximum dimension of problems to be solved without using Sequential
           Random Embeddings
    :param n_reduced_dims_eff: [int] Number of effective reduced dimensions used by the Sequential
           Random Embeddings algorithm
    :return: Initial conditions for all bacteria [np.array of shape (n_bacteria, n_dims)]
    """

    x0 = np.array(x0)
    if len(x0.shape) == 1:
        n_dims = x0.shape[0]
        n_bacteria = (n_bacteria_per_dim ** n_dims if n_dims <= max_dims else
                      n_bacteria_per_dim ** n_reduced_dims_eff)
        x0_population = np.tile(x0, (n_bacteria, 1))
    elif len(x0.shape) == 2:
        n_dims = x0.shape[1]
        n_bacteria = x0.shape[0]
        n_bacteria_target = (n_bacteria_per_dim ** n_dims if n_dims <= max_dims else
                             n_bacteria_per_dim ** n_reduced_dims_eff)
        if n_bacteria != n_bacteria_target:
            warnings.warn('The number of bacteria given by x0 does not match the number of ' +
                          'bacteria given by the relation ' +
                          'n_bacteria = n_bacteria_per_dim ** n_dims if n_dims <= max_dims else ' +
                          'n_bacteria_per_dim ** (n_reduced_dims + 1). The latter implies that ' +
                          f'n_bacteria = {n_bacteria_target}, whereas the former implies ' +
                          f'that n_bacteria = {n_bacteria}. Using n_bacteria = {n_bacteria}.')
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

    return np.pad(trace, ((0, paddingLength), (0, 0)), mode="edge")


def _sequential_random_embeddings(f, x0, bounds, n_reduced_dims_eff=3, n_embeddings=10,
                                  verbosity=1, **optimizer_kwargs):
    """
    Implementation of the Sequential Random Embeddings algorithm described in
    +++++
    H. Qian, Y.-Q. Hu, and Y. Yu, Derivative-Free Optimization of High-Dimensional Non-Convex
    Functions by Sequential Random Embeddings, Proceedings of the Twenty-Fifth International Joint
    Conference on Artificial Intelligence, AAAI Press (2016).
    +++++
    The idea is basically to reduce high-dimensional problems to low-dimensional ones by embedding
    the original, high-dimensional search space ℝ^h into a low dimensional one, ℝ^l, by
    sequentially applying the random linear transformation
    x(n+1) = α(n+1)x(n) + A•y(n+1),    x ∈ ℝ^h, y ∈ ℝ^l, A ∈ N(0, 1)^(h×l), α ∈ ℝ
    and minimizing the objective function f(αx + A•y) w.r.t. (α, y).

    :param f: [callable] Objective function. Must accept its argument x as numpy array
    :param x0: [np.array] Initial values for the bacteria population in the original,
           high-dimensional space ℝ^h
    :param bounds: [callable] Bounds projection, see description of parameter
           ``projection_callback`` in :func:`local_search.bfgs_b`
    :param n_reduced_dims_eff: [int] Effective dimension of the embedded problem, ℝ^(l+1)
    :param n_embeddings: [int] Number of embedding iterations
    :param verbosity: [int] Output verbosity. Must be 0, 1, or 2
    :param optimizer_args: [dict] Arguments to pass to the actual optimization routine
    :return: Best minimum of f found [scipy.optimize.OptimizeResult]
    """

    assert verbosity in [0, 1, 2], 'verbosity must be 0, 1, or 2.'

    orig_dim = x0.shape[1]
    x = np.zeros(orig_dim)
    x_best = x.copy()
    f_best = np.inf
    nfev = nit = 0
    success_best = False
    for i in range(n_embeddings):
        A = np.random.normal(size=(orig_dim, n_reduced_dims_eff - 1))

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
        y0 = np.zeros((x0.shape[0], n_reduced_dims_eff))
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
    Metaheuristic global optimization algorithm combining a bacterial run-and-tumble chemotactic
    search with a local, gradient-based search around the best minimum candidate points.
    The algorithm's goal is to find
                                        min f(x), x ∈ Ω,
    where f: Ω ⊂ ℝ^n → ℝ.
    Since the chemotactic search becomes more and more ineffective with increasing problem
    dimensionality, Sequential Random Embeddings are used to solve the optimization problem once its
    dimensionality exceeds a given threshold.

    :param f: [callable] Objective function. Must accept its argument x as numpy array
    :param x0: [array-like object] Optional initial conditions object. Must have the shape
           (n_bacteria, n_dims) or (n_dims,). If x0 == None, initial conditions are sampled randomly
           or uniformly-spaced from Ω. Note that this only works if Ω is a rectangular box, i.e., if
           no or non-rectangular bounds are imposed, x0 must not be None
    :param bounds: [callable or array-like object] Defines the bounded domain Ω. If provided, must
           be one of the following:
           - Bounds projection callback, as defined in description of parameter
             ``projection_callback`` in :func:`local_search.bfgs_b`
           - Rectangular box constraints. For each component x_i of x,
             bounds[i, 0] <= x_i <= bounds[i, 1], that is, bounds must have shape (n_dims, 2)
    :param domain_scale: [float] Scale of the optimization problem. If not provided, the algorithm
           tries to guess the scale from any provided rectangular box constraints. Used for
           auto-scaling algorithm stepsizes
    :param init: [string] Determines how initial bacteria positions are sampled from Ω if
           x0 == None, see description of parameter ``x0``. Currently supported: 'random' and
           'uniform'
    :param stepsize_start: [float] See description of parameter ``stepsize_start`` in
           :func:`global_search.run_and_tumble`. If not provided, the algorithm tries to auto-scale
           this length to the problem's scale
    :param stepsize_decay_fac: [float] Factor by which the run-and-tumble stepsize has decayed in
           the last run-and-tumble iteration compared to its initial value
    :param base_tumble_rate: [float] See description of parameter ``base_tumble_rate`` in
           :func:`global_search.run_and_tumble`
    :param niter_rt: [int] Maximum number of run-and-tumble iterations
    :param n_bacteria_per_dim: [int] How many bacteria to spawn in each dimension. Note that the
           total number of bacteria is
           i)  n_bacteria = n_bacteria_per_dim ** n_dims if n_dims <= max_dims or
           ii) n_bacteria = n_bacteria_per_dim ** (n_reduced_dims + 1) if n_dims > max_dims.
           If x0 is provided with shape (n_bacteria, n_dims), n_bacteria should agree with this
           relation.
    :param stationarity_window: [int] See description of parameter ``stationarity_window`` in
           :func:`global_search.run_and_tumble`
    :param eps_stat: [float] See description of parameter ``stationarity_window`` in
           :func:`global_search.run_and_tumble`
    :param attraction: [bool] See description of parameter ``attraction`` in
           :func:`global_search.run_and_tumble`
    :param attraction_window: [int] See description of parameter ``attraction_window`` in
           :func:`global_search.run_and_tumble`
    :param attraction_sigma: [float] See description of parameter ``attraction_sigma`` in
           :func:`global_search.run_and_tumble`. If not provided, the algorithm tries to auto-scale
           this length to the problem's scale
    :param attraction_strength: [float] See description of parameter ``attraction_strength`` in
           :func:`global_search.run_and_tumble`
    :param bounds_reflection: [bool] See description of parameter ``bounds_reflection`` in
           :func:`global_search.run_and_tumble`
    :param n_best_selection: [int] At the end of the run-and-tumble exploration stage, a local
           gradient-based search is performed, starting from the best positions found thus far by
           the n_best_selection best bacteria
    :param c_gd: [float] See description of parameter ``c`` in :func:`local_search.bfgs_b`
    :param a_gd: [float] See description of parameter ``a`` in :func:`local_search.bfgs_b`. If not
           provided, the algorithm tries to auto-scale this length to the problem's scale
    :param n_linesearch_gd: [int] See description of parameter ``n_linesearch`` in
           :func:`local_search.bfgs_b`
    :param alpha_linesearch_gd: [float] See description of parameter ``alpha_linesearch`` in
           :func:`local_search.bfgs_b`
    :param beta_linesearch_gd: [float] See description of parameter ``beta_linesearch`` in
           :func:`local_search.bfgs_b`
    :param eps_abs_gd: [float] See description of parameter ``eps_abs`` in
           :func:`local_search.bfgs_b`
    :param eps_rel_gd: [float] See description of parameter ``eps_rel`` in
           :func:`local_search.bfgs_b`
    :param niter_gd: [int] Maximum number of local, gradient-based search iterations
    :param n_embeddings: [int] Number of embedding iterations when using Sequential Random
           Embeddings. Only has an effect if n_dims > max_dims
    :param max_dims: [int] Maximum dimension of problems to be solved without using Sequential
           Random Embeddings
    :param n_reduced_dims: [int] Dimension of the embedded problem. Only has an effect if
           n_dims > max_dims
    :param verbosity: [int] Output verbosity. Must be 0, 1, or 2
    :return: Best minimum of f found [scipy.optimize.OptimizeResult]
    """

    assert verbosity in [0, 1, 2], 'verbosity must be 0, 1, or 2.'
    assert n_reduced_dims >= 2, 'n_reduced_dims must not be less than 2.'

    n_reduced_dims_eff = n_reduced_dims + 1

    if bounds is None or callable(bounds):
        assert x0 is not None, ('If no box constraints are provided for bounds, x0 must not be ' +
                                'None.')
        x0_population = _prepare_x0(x0, n_bacteria_per_dim, max_dims, n_reduced_dims_eff)
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
            x0_population = _prepare_x0(x0, n_bacteria_per_dim, max_dims, n_reduced_dims_eff)
            n_bacteria, n_dims = x0_population.shape
            bound_lower, bound_upper = _prepare_bounds(bounds, n_dims)
        else:
            bound_lower, bound_upper = _prepare_bounds(bounds, None)
            n_dims = len(bound_lower)
            n_bacteria = (n_bacteria_per_dim ** n_dims if n_dims <= max_dims else
                          n_bacteria_per_dim ** n_reduced_dims_eff)
            if init == 'uniform' and n_dims > max_dims:
                init = 'random'
                if verbosity > 0:
                    warnings.warn('The option init="uniform" is only available for problems with ' +
                                  'dimensionality less than or equal to max_dims, which was ' +
                                  f'set to {max_dims}. Since the current problem has ' +
                                  f'dimensionality {n_dims}, init was automatically set to ' +
                                  f'"random".')
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
                                             n_reduced_dims_eff=n_reduced_dims_eff,
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
                                             niter_gd=niter_gd,
                                             max_dims=n_reduced_dims_eff)

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

            local_optimization_result = bfgs_b(f,
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
            x_best_gd[n] = local_optimization_result.x
            f_min_gd[n] = local_optimization_result.f
            nfev_gd += local_optimization_result.nfev
            nit_gd += local_optimization_result.nit
            nit_gd_arr[n] = local_optimization_result.nit
            success_gd[n] = local_optimization_result.success
            trace_gd[:, sortIdx[n], :] = _pad_trace(local_optimization_result.trace, niter_gd)

        result = OptimizeResult()
        result.success = success_gd.any()
        result.x = x_best_gd[np.argmin(f_min_gd)]
        result.fun = np.min(f_min_gd)
        result.nfev = nfev + nfev_gd
        result.nit = nit + nit_gd
        trace_gd = trace_gd[:np.max(nit_gd_arr).astype(int)]
        result.trace = np.concatenate((trace, trace_gd))

        return result
