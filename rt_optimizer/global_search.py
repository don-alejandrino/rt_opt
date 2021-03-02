import numpy as np
from scipy.stats import stats


def run_and_tumble(f, x0_population, projection_callback, niter, stepsize_start, stepsize_end,
                   base_tumble_rate=0.1, stationarity_window=20, eps_stat=1e-3, attraction=False,
                   attraction_window=10, attraction_sigma=1, attraction_strength=0.5,
                   bounds_reflection=False, verbosity=1):
    """
    ToDo: Write docstring

    :param f:
    :param x0_population:
    :param projection_callback:
    :param niter:
    :param stepsize_start:
    :param stepsize_end:
    :param base_tumble_rate:
    :param stationarity_window:
    :param eps_stat:
    :param attraction:
    :param attraction_window:
    :param attraction_sigma:
    :param attraction_strength:
    :param bounds_reflection:
    :param verbosity:
    :return:
    """

    n_bacteria = x0_population.shape[0]
    n_dims = x0_population.shape[1]

    x = x0_population.copy()
    x_best = x0_population.copy()
    x_old = x0_population.copy()
    f_old = np.array([f(val) for val in x_old])
    f_best = f_old.copy()
    f_max = f_old.copy()
    nfev = n_bacteria
    trace = np.empty((niter + 1, n_bacteria, n_dims))
    trace[0] = x0_population.copy()
    x_sum = x0_population.sum(axis=0)
    x_mean_history = []

    v = np.empty(x0_population.shape)
    for m in range(n_bacteria):
        v_m = np.random.uniform(-1, 1, n_dims)
        while not v_m.any():
            v_m = np.random.uniform(-1, 1, n_dims)
        v_m = v_m / np.sqrt(np.sum(v_m ** 2))
        v[m] = v_m

    for n in range(niter):
        alpha = stepsize_start + (stepsize_end - stepsize_start) * (n ** 2) / (niter ** 2)

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
        x, bounds_hit = projection_callback(x)
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
        # Calculate new orientation
        for m, tr in enumerate(tumble_rate):
            if bounds_reflection and bounds_hit[m].any():
                # Reflection at boundaries
                v[m] = -v[m]
            elif bounds_hit[m].any() or np.random.uniform() > 1 - tr:
                # Realistically, tr must be clipped to [0, 1]. However, the inequality above is not
                # influenced by this clipping and we thus omit it in order to save computation time
                v_m = np.random.uniform(-1, 1, n_dims)
                while not v_m.any():
                    v_m = np.random.uniform(-1, 1, n_dims)
                v_m = v_m / np.sqrt(np.sum(v_m ** 2))
                v[m] = v_m

        # Remember best results
        x_best = np.where((f_new < f_best)[:, None], x, x_best)
        f_best = np.minimum(f_new, f_best)
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
                if verbosity > 0:
                    print(f'Run-and-tumble stage: Stationary state detected after {nit} steps.')
                break

    else:
        if verbosity > 0:
            print('Run-and-tumble stage: No stationary state could be detected after ' +
                  f'{niter + 1} iterations. Please try increasing niter or the stationarity ' +
                  f'detection threshold eps_stat.')
        nit = niter + 1

    trace = trace[:(nit + 1)]

    return x_best, f_best, nfev, nit, trace
