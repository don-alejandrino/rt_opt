import warnings

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.stats import stats


def prepare_bounds(bounds, n_dims):
    """
    ToDo: Write docstring

    :param bounds:
    :param n_dims:
    :return:
    """

    if bounds is not None:
        bounds = np.array(bounds)
        assert bounds.shape == (n_dims, 2), ('bounds has wrong shape. Expected shape: ' +
                                             '(n_dims, 2), where n_dims is the dimensionality of ' +
                                             'the problem.')
        bound_lower = bounds[:, 0]
        bound_upper = bounds[:, 1]
        assert (bound_upper > bound_lower).all(), ('Upper bound must always be larger than lower ' +
                                                   'bound.')
        return bound_lower, bound_upper
    else:
        return None, None


def padTrace(trace, targetLength):
    """
    Pad single-bacteria trace to given length.

    :param trace: [np.array] Single-bacteria trace
    :param targetLength: [int] Desired length
    :return: Padded trace [np.array]
    """

    currentLength = trace.shape[0]
    paddingLength = (targetLength - currentLength)

    return np.pad(trace, [(0, paddingLength), (0, 0)], mode="edge")


def run_and_tumble(f, x0, bounds=None, alpha_start=None, alpha_decay_fac=1e-3, base_tumble_rate=0.1,
                   stationarity_window=20, eps_stat=1e-2, n_best_selection=5, c_spsa=1e-6,
                   a_spsa=None, gamma_spsa=0.101, alpha_spsa=0.602, A_fac_spsa=0.05,
                   beta_1_adam=0.9, beta_2_adam=0.9, eps_gd=1e-10, niter=500, n_bacteria=10,
                   attraction=False, attraction_window=10, attraction_sigma=None,
                   attraction_strength=0.5, verbosity=0):
    """
    ToDo: Write docstring

    :param f:
    :param x0:
    :param bounds:
    :param alpha_start:
    :param alpha_decay_fac:
    :param base_tumble_rate:
    :param stationarity_window:
    :param eps_stat:
    :param n_best_selection:
    :param c_spsa:
    :param a_spsa:
    :param gamma_spsa:
    :param alpha_spsa:
    :param A_fac_spsa:
    :param beta_1_adam:
    :param beta_2_adam:
    :param eps_gd:
    :param niter:
    :param n_bacteria:
    :param attraction:
    :param attraction_window:
    :param attraction_sigma:
    :param attraction_strength:
    :param verbosity:
    :return:
    """

    assert niter > stationarity_window, 'niter must be larger than stationarity_window.'
    assert verbosity in [0, 1, 2], 'verbosity must be 0, 1, or 2.'
    assert n_best_selection <= n_bacteria, 'n_best_selection must be less equal than n_bacteria.'

    result = OptimizeResult()
    x0 = np.array(x0)
    if len(x0.shape) == 1:
        n_dims = x0.shape[0]
        x0_population = np.tile(x0, (n_bacteria, 1))
    elif len(x0.shape) == 2 and x0.shape[0] == 2:
        n_dims = x0.shape[1]
        assert (x0[1] > x0[0]).all(), 'x0[1, i] > x0[0, i] must hold for all i.'
        x0_population = np.random.uniform(x0[0], x0[1], size=(n_bacteria, n_dims))
    else:
        raise ValueError('x0 must be either a matrix of shape (2, n_dims) or a vector of size ' +
                         'n_dims.')

    bound_lower, bound_upper = prepare_bounds(bounds, n_dims)

    if alpha_start is not None:
        auto_scale_alpha = False
    else:
        auto_scale_alpha = True
        alpha_start = 1e-1
    alpha_end = alpha_decay_fac * alpha_start

    if attraction_sigma is not None:
        auto_scale_sigma = False
    else:
        auto_scale_sigma = True
        attraction_sigma = 1

    if a_spsa is not None:
        auto_scale_a = False
    else:
        auto_scale_a = True
        a_spsa = 1e-2

    x0_population_orig = x0_population.copy()
    if bounds is not None:
        x0_population = np.maximum(x0_population, bound_lower)
        x0_population = np.minimum(x0_population, bound_upper)
    if not np.array_equal(x0_population, x0_population_orig):
        warnings.warn('Found initial conditions outside the defined search domain.')

    v = np.empty(x0_population.shape)
    for m in range(n_bacteria):
        v_m = np.random.uniform(-1, 1, n_dims)
        while not v_m.any():
            v_m = np.random.uniform(-1, 1, n_dims)
        v_m = v_m / np.sqrt(np.sum(v_m ** 2))
        v[m] = v_m

    max_scale = None
    if bounds is not None:
        domain_range = bound_upper - bound_lower
        max_scale = np.max(np.where(np.isinf(domain_range), 0, domain_range))
    elif len(x0_population.shape) == 2 and x0_population.shape[1] == 2:
        domain_range = x0[1] - x0[0]
        max_scale = np.max(np.where(np.isinf(domain_range), 0, domain_range))
    if max_scale is not None and max_scale > 0:
        if auto_scale_alpha:
            alpha_start = alpha_start * max_scale
            alpha_end = alpha_end * max_scale
        if auto_scale_sigma:
            attraction_sigma = attraction_sigma * max_scale
        if auto_scale_a:
            a_spsa = a_spsa * max_scale

    x = x0_population.copy()
    x_best = x0_population.copy()
    x_old = x0_population.copy()
    f_old = np.array([f(val) for val in x_old])
    f_min = f_old.copy()
    f_max = f_old.copy()
    nfev = n_bacteria
    trace = np.empty((niter + 1, n_bacteria, n_dims))
    trace[0] = x0_population.copy()
    x_sum = x0_population.sum(axis=0)
    x_mean_history = []

    for n in range(niter):
        alpha = alpha_start + (alpha_end - alpha_start) * (n ** 2) / (niter ** 2)

        if attraction:
            kernel = (x[:, None, None, :] -
                      trace[None, (n + 1 - min(n, attraction_window)):(n + 1), :, :])
            grad_attractant = (
                    attraction_strength / 2 / (attraction_sigma ** 2) * kernel *
                    np.exp(-(np.square(kernel) / 2 /
                             (attraction_sigma ** 2)).sum(axis=3))[:, :, :, None]
            ).sum(axis=(1, 2))
        else:
            grad_attractant = np.zeros(x.shape)

        # Run
        x = x_old + (v - grad_attractant) * alpha
        if bounds is not None:
            x = np.maximum(x, bound_lower)
            x = np.minimum(x, bound_upper)
        boundary_hit = np.where(((x == bound_lower) | (x == bound_upper)), True, False)
        f_new = np.array([f(val) for val in x])
        nfev += n_bacteria
        trace[n + 1] = x.copy()

        # Tumble
        # We add a small constant (1e-9) to the denominator below, in order to account for the fact
        # that a bacterium may be stuck at the boundary, in which case x[i, :] = x_old[i, :]
        delta_f = (f_new - f_old) / (np.sqrt(np.sum((x - x_old) ** 2, axis=1)) + 1e-9)
        # Avoid exp over/underflow
        delta_f = np.maximum(np.minimum(delta_f, 100), -100)
        tumble_rate = base_tumble_rate * np.exp(delta_f)
        # Realistically, tumble_rate must be clipped to [0, 1]. However, the inequality below is not
        # influenced by this clipping and we thus omit it in order to save computation time
        for m, tr in enumerate(tumble_rate):
            if boundary_hit[m].any() or np.random.uniform() > 1 - tr:
                v_m = np.random.uniform(-1, 1, n_dims)
                while not v_m.any():
                    v_m = np.random.uniform(-1, 1, n_dims)
                v_m = v_m / np.sqrt(np.sum(v_m ** 2))
                v[m] = v_m

        # Remember best results
        x_best = np.where((f_new < f_min)[:, None], x, x_best)
        f_min = np.minimum(f_new, f_min)
        f_max = np.maximum(f_new, f_max)
        x_old = x.copy()
        f_old = f_new.copy()

        if verbosity == 2:
            for m in range(n_bacteria):
                print('Run-and-tumble step {}, bacterium {}:\tx = {}, f(x) = {}'
                      .format(n + 1, m, x[m], f_new[m]))

        x_sum = x_sum + x.sum(axis=0)
        x_mean = x_sum / n_bacteria / (n + 1)
        x_mean_history.append(x_mean)
        if (n + 1) % stationarity_window == 0:
            window = np.array(x_mean_history[-stationarity_window:]).sum(axis=1)
            slope, intercept, rValue, _pValue, _stdErr = stats.linregress(
                np.linspace(0, 1, len(window)), window
            )
            if rValue ** 2 > 0.9 and abs(slope / intercept) < eps_stat:
                nit = n + 1
                success_rt = True
                if verbosity > 0:
                    print(f'Stationary state detected after {nit} steps.')
                break

    else:
        if verbosity > 0:
            print(f'No stationary state could be detected after {niter + 1} iterations. ' +
                  'Please try increasing niter or the stationarity detection threshold eps_stat.')
        nit = niter + 1
        success_rt = False

    if verbosity == 2:
        print('===================================================================================')
    if verbosity > 0:
        print(f'Best result thus far is x = {x_best[np.argmin(f_min)]}, f(x) = {np.min(f_min)}. '
              'Starting SPSA gradient descent.')

    trace = trace[:(nit + 1)]
    sortIdx = f_min.argsort()
    x_best_selection = x_best[sortIdx[:n_best_selection]]
    x_best_spsa = np.empty(x_best_selection.shape)
    f_min_spsa = np.empty(n_best_selection)
    nfev_spsa = 0
    nit_spsa = 0
    success_spsa = True
    trace_spsa = np.empty((niter, n_bacteria, n_dims))
    trace_spsa[:, sortIdx[n_best_selection:], :] = trace[-1, sortIdx[n_best_selection:], :]
    nit_spsa_arr = np.empty(n_best_selection)

    for n, x0 in enumerate(x_best_selection):
        if verbosity == 2:
            print(f'Performing SPSA gradient descent for bacterium {n}.')
        (x_spsa_single,
         f_min_spsa_single,
         nfev_spsa_single,
         nit_spsa_single,
         success_spsa_single,
         trace_spsa_single) = adam_spsa(f, x0, bounds=bounds, c=c_spsa, a=a_spsa, gamma=gamma_spsa,
                                        alpha=alpha_spsa, A_fac=A_fac_spsa, beta_1=beta_1_adam,
                                        beta_2=beta_2_adam, eps=eps_gd, niter=niter,
                                        verbosity=verbosity)
        x_best_spsa[n] = x_spsa_single
        f_min_spsa[n] = f_min_spsa_single
        nfev_spsa += nfev_spsa_single
        nit_spsa += nit_spsa_single
        nit_spsa_arr[n] = nit_spsa_single
        if not success_spsa_single:
            success_spsa = False
        trace_spsa[:, sortIdx[n], :] = padTrace(trace_spsa_single, niter)

    if success_rt:
        result.success = success_spsa
    else:
        result.success = False
    result.x = x_best_spsa[np.argmin(f_min_spsa)]
    result.fun = np.min(f_min_spsa)
    result.nfev = nfev + nfev_spsa
    result.nit = nit + nit_spsa
    trace_spsa = trace_spsa[:np.max(nit_spsa_arr).astype(int)]
    result.trace = np.concatenate((trace, trace_spsa))

    return result


def adam_spsa(f, x0, bounds=None, c=1e-3, a=1e-3, gamma=0.101, alpha=0.602, A_fac=0.05, beta_1=0.9,
              beta_2=0.9, eps=1e-12, niter=10000, verbosity=0):
    """
    ToDo: Write docstring

    :param f:
    :param x0:
    :param bounds:
    :param c:
    :param a:
    :param gamma:
    :param alpha:
    :param A_fac:
    :param beta_1:
    :param beta_2:
    :param eps:
    :param niter:
    :param verbosity:
    :return:
    """

    assert verbosity in [0, 1, 2], 'verbosity must be 0, 1, or 2.'

    x0 = np.array(x0)
    n_dims = len(x0)
    A = A_fac * niter
    m = v = 0

    bound_lower, bound_upper = prepare_bounds(bounds, n_dims)

    trace = np.empty((niter, n_dims))
    f0 = f(x0)
    nfev = 1
    f_best = f0
    x_best = x0.copy()
    x = x0.copy()
    for k in range(niter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = 2 * np.round(np.random.uniform(0, 1, n_dims)) - 1

        # Boundary hit
        if (x == bound_lower).any() or (x == bound_upper).any():
            boundary_stuck = np.zeros(n_dims, dtype=bool)
            idcs = np.argwhere((x == bound_lower) | (x == bound_upper))
            for i in idcs:
                delta_i = np.zeros(n_dims)
                delta_i[i] = delta[i]
                f_minus = f(x - ck * delta_i)
                f_plus = f(x + ck * delta_i)
                nfev += 2
                if ((f_plus - f_minus <= 0 and x[i] == bound_upper[i]) or
                        (f_plus - f_minus >= 0 and x[i] == bound_upper[i])):
                    boundary_stuck[i] = True
            delta = np.where(boundary_stuck, 0, delta)

        f_minus = f(x - ck * delta)
        f_plus = f(x + ck * delta)
        nfev += 2
        ghat = (f_plus - f_minus) / (2 * ck * np.where(delta == 0, np.inf, delta))

        # Adam algorithm, with the true gradient replaced by the SPSA
        m = beta_1 * m + (1 - beta_1) * ghat
        v = beta_2 * v + (1 - beta_2) * np.power(ghat, 2)
        m_hat = m / (1 - np.power(beta_1, k + 1))
        v_hat = v / (1 - np.power(beta_2, k + 1))
        x = x - ak * m_hat / (np.sqrt(v_hat) + 1e-9)

        if bounds is not None:
            x = np.maximum(x, bound_lower)
            x = np.minimum(x, bound_upper)
        f_new = f(x)
        if f_new <= f_best:
            f_best = f_new
            x_best = x.copy()
        else:
            if verbosity == 2:
                print(f'Reducing a to {0.95 * a}.')
            x = x_best.copy()
            a *= 0.95

        trace[k] = x.copy()
        if verbosity == 2:
            print('SPSA step {}:\tx = {}, ghat = {}'.format(k + 1, x, ghat))

        if abs(f_plus - f_minus) < eps:
            nit = k + 1
            success = True
            if verbosity > 0:
                print(f'SPSA Gradient descent target accuracy reached after {nit} steps.')
            break

    else:
        if verbosity > 0:
            print(f'Could not reach desired SPSA gradient descent accuracy after {niter + 1} ' +
                  'iterations. Please try increasing niter or the SPSA gradient descent accuracy ' +
                  'threshold.')
        nit = niter + 1
        success = False
    f_final = f(x)
    trace = trace[:nit]
    nfev += 1

    return x, f_final, nfev, nit, success, trace
