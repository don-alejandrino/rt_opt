"""Algorithms for global minimum search."""

import logging
from warnings import warn

import numpy as np
from scipy.stats import linregress

from rt_opt.config.global_search import RunAndTumbleConfig
from rt_opt.search.search_output import MultiSearchOutput
from rt_opt.utils.types import ObjectiveFunctionType, ProjectionCallbackType

logger = logging.getLogger(__name__)


def run_and_tumble(  # noqa: PLR0915
    f: ObjectiveFunctionType,
    x0_population: np.ndarray,
    projection_callback: ProjectionCallbackType,
    config: RunAndTumbleConfig | None = None,
) -> MultiSearchOutput:
    """Bacterial run-and-tumble global search algorithm.

    Implementation of a bacterial run-and-tumble optimizer algorithm, motivated by the
    chemotactic behavior of E. coli. The motion of E. coli consists of directed,
    ballistic "runs", interrupted by sudden random re-orientations, so-called "tumbles",
    that appear at some given rate. If a bacterium detects to swim toward higher
    concentrations of an attractant (i.e., if the attractant concentration increases
    during a run), its tumbling rate is lowered, thus inducing an effective movement
    toward the attractant's source.
    Here, we implement a simplified E. coli chemotaxis model, where the attractant
    concentration is the negative of a given objective function.

    :param f: [callable] Objective function. Must accept its argument x as numpy array.
    :param x0_population: [np.array] Initial condition for the bacteria population.
           x0_population.shape[0] defines the number of bacteria and
           x0_population.shape[1] the problem dimensionality.
    :param projection_callback: [callable] Bounds projection, see description of
           parameter `projection_callback` in :func:`search.local_search.bfgs_b`.
    :param config: Configuration provided to the run-and-tumble algorithm.
    :return: A `MultiSearchOutput` instance containing the search results.

    """
    if config is None:
        config = RunAndTumbleConfig()

    if config.stepsize_start is None:
        warn(
            "`stepsize_start` was not provided. As auto-scaling of"
            "`stepsize_start` is not implemented for the low-level search routines, it "
            "has been set to 0.1 by default. Please consider tuning this parameter to "
            "your specific problem.",
            stacklevel=1,
        )
        stepsize_start = 0.1
    else:
        stepsize_start = config.stepsize_start

    if config.attraction_sigma is None:
        warn(
            "`attraction_sigma` was not provided. As auto-scaling of "
            "`attraction_sigma` is not implemented for the low-level search routines, "
            "it has been set to 1 by default. Please consider tuning this parameter to "
            "your specific problem.",
            stacklevel=1,
        )
        attraction_sigma = 1
    else:
        attraction_sigma = config.attraction_sigma

    if config.niter <= config.stationarity_window:
        err_msg = "`niter` must be larger than `stationarity_window`."
        raise ValueError(err_msg)

    rng = np.random.default_rng(config.seed)

    n_bacteria = x0_population.shape[0]
    n_dims = x0_population.shape[1]

    x = x0_population.copy()
    x_best = x0_population.copy()
    x_old = x0_population.copy()
    f_old = np.array([f(val) for val in x_old])
    f_best = f_old.copy()
    nfev = n_bacteria
    trace = np.empty((config.niter + 1, n_bacteria, n_dims))
    trace[0] = x0_population.copy()
    x_sum = x0_population.sum(axis=0)
    x_mean_history = []
    success = False

    # Initial random bacteria orientations
    v = _initialize_bacteria_orientations(n_bacteria, n_dims, rng)

    stepsize_end = stepsize_start * config.stepsize_decay_fac
    niter = config.niter
    base_tumble_rate = config.base_tumble_rate
    stationarity_window = config.stationarity_window
    stationarity_r_value_threshold = config.stationarity_r_value_threshold
    eps_stat = config.eps_stat
    attraction = config.attraction
    attraction_window = config.attraction_window
    attraction_strength = config.attraction_strength
    bounds_reflection = config.bounds_reflection

    for n in range(niter):
        alpha = stepsize_start + (stepsize_end - stepsize_start) * (n**2) / (niter**2)

        grad_attractant = (
            _calculate_attractant_gradient(
                x,
                trace,
                n,
                attraction_window,
                attraction_sigma,
                attraction_strength,
            )
            if attraction
            else np.zeros(x.shape)
        )

        # Run
        x = x_old + (v - grad_attractant) * alpha
        x, bounds_hit = projection_callback(x)
        f_new = np.array([f(val) for val in x])
        nfev += n_bacteria
        trace[n + 1] = x.copy()

        # Tumble
        # We add a small constant (1e-9) to the denominator below, in order to account
        # for the fact that a bacterium may be stuck at the boundary, in which case
        # x[i, :] = x_old[i, :].
        delta_f = (f_new - f_old) / (np.sqrt(np.sum((x - x_old) ** 2, axis=1)) + 1e-9)

        # Avoid exp over/underflow
        delta_f = np.maximum(np.minimum(delta_f, 100), -100)
        tumble_rate = base_tumble_rate * np.exp(delta_f)

        _tumble(v, tumble_rate, bounds_hit, bounds_reflection, n_dims, rng)

        # Remember best results
        x_best = np.where((f_new < f_best)[:, None], x, x_best)
        f_best = np.minimum(f_new, f_best)
        x_old = x.copy()
        f_old = f_new.copy()

        if logger.isEnabledFor(logging.DEBUG):
            for m in range(n_bacteria):
                logger.debug(
                    "Run-and-tumble step %d, bacterium %d:\tx = %s, f(x) = %g",
                    n + 1,
                    m,
                    str(x[m]),
                    f_new[m],
                )

        # Calculate mean position of the bacteria
        x_sum = x_sum + x.sum(axis=0)
        x_mean = x_sum / n_bacteria / (n + 1)
        x_mean_history.append(x_mean)
        if (n + 1) % stationarity_window == 0:
            # If the mean position has had a relative change less than `eps_stat` over a
            # step window `stationarity_window`, we consider the bacteria distribution
            # as stationary.
            window = np.array(x_mean_history[-stationarity_window:]).sum(axis=1)
            regression_result = linregress(np.linspace(0, 1, len(window)), window)
            if (
                regression_result.rvalue**2 > stationarity_r_value_threshold
                and abs(regression_result.slope / regression_result.intercept)
                < eps_stat
            ):
                nit = n + 1
                logger.info(
                    "Run-and-tumble stage: Stationary state detected after %d steps.",
                    nit,
                )
                success = True
                break

    else:
        logger.info(
            "Run-and-tumble stage: No stationary state could be detected after "
            "%d iterations. If you want to run the run-and-tumble stage until"
            "stationarity, please try increasing niter or the stationarity detection "
            "threshold eps_stat.",
            niter,
        )
        nit = niter

    trace = trace[: (nit + 1)]

    return MultiSearchOutput(x_best, f_best, nfev, nit, success, trace)


def _initialize_bacteria_orientations(
    n_bacteria: int,
    n_dims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    v = np.empty((n_bacteria, n_dims))
    for m in range(n_bacteria):
        v_m = rng.uniform(-1, 1, n_dims)
        while not v_m.any():
            v_m = rng.uniform(-1, 1, n_dims)
        v_m = v_m / np.sqrt(np.sum(v_m**2))
        v[m] = v_m
    return v


def _calculate_attractant_gradient(  # noqa: PLR0913
    x: np.ndarray,
    trace: np.ndarray,
    n: int,
    attraction_window: int,
    attraction_sigma: float,
    attraction_strength: float,
) -> np.ndarray:
    # Calculate attraction between the bacteria traces
    kernel = (
        x[:, None, None, :]
        - trace[None, (n + 1 - min(n, attraction_window)) : (n + 1), :, :]
    )
    return (
        attraction_strength
        / 2
        / (attraction_sigma**2)  # type: ignore[reportOptionalOperand]
        * kernel
        * np.exp(
            -(
                np.square(kernel)
                / 2
                / (
                    attraction_sigma**2  # type: ignore[reportOptionalOperand]
                )
            ).sum(axis=3)
        )[:, :, :, None]
    ).sum(axis=(1, 2))


def _tumble(  # noqa: PLR0913
    v: np.ndarray,
    tumble_rate: np.ndarray,
    bounds_hit: np.ndarray,
    bounds_reflection: bool,
    n_dims: int,
    rng: np.random.Generator,
) -> None:
    # Calculate new orientation
    for m, tr in enumerate(tumble_rate):
        if bounds_reflection and bounds_hit[m].any():
            # Reflection at boundaries
            v[m] = -v[m]
        elif bounds_hit[m].any() or rng.uniform() > 1 - tr:
            # Realistically, `tr` must be clipped to [0, 1]. However, the inequality
            # above is not influenced by this clipping, and we thus omit it in order
            # to save computation time.
            v_m = rng.uniform(-1, 1, n_dims)
            while not v_m.any():
                v_m = rng.uniform(-1, 1, n_dims)
            v_m = v_m / np.sqrt(np.sum(v_m**2))
            v[m] = v_m
