"""Optimizer entrypoint module."""

import itertools
import logging
import warnings

import numpy as np
from numpy._typing import ArrayLike
from scipy import spatial, special
from scipy.optimize import OptimizeResult

from rt_opt.config.global_search import RunAndTumbleConfig
from rt_opt.config.local_search import BFGSBConfig
from rt_opt.config.optimizer import OptimizationConfig
from rt_opt.dimension_reduction.sre import sequential_random_embeddings
from rt_opt.search.global_search import run_and_tumble
from rt_opt.search.local_search import bfgs_b
from rt_opt.search.search_output import SingleSearchOutput
from rt_opt.utils.io import pad_trace, prepare_bounds, prepare_x0
from rt_opt.utils.types import ObjectiveFunctionType, ProjectionCallbackType

logger = logging.getLogger(__name__)


class Optimizer:
    """Main optimizer class doing the heavy lifting part of the `rt_opt` routine.

    Implements the OptimizerProtocol.
    """

    def __init__(
        self,
        global_search_config: RunAndTumbleConfig,
        n_best_selection: int,
        local_search_config: BFGSBConfig,
    ) -> None:
        """Initialize the optimizer.

        :param global_search_config: Configuration for the global search stage.
        :param n_best_selection: At the end of the global search stage, a local,
               gradient-based search is performed, starting from the best positions
               found thus far by the `n_best_selection` best bacteria.
        :param local_search_config: Configuration for the local search stage.

        """
        self.global_search_config = global_search_config
        self.n_best_selection = n_best_selection
        self.local_search_config = local_search_config

    def run(
        self,
        f: ObjectiveFunctionType,
        x0_population: np.ndarray,
        projection_callback: ProjectionCallbackType,
        projection_callback_population: ProjectionCallbackType,
    ) -> SingleSearchOutput:
        """Run the optimization routine.

        :param f: Objective function. Must accept its argument `x` as numpy array.
        :param x0_population: Initial condition. Must have the shape
               (n_bacteria, n_dims).
        :param projection_callback: Bounds projection callback, see description of
               parameter `projection_callback` in :func:`search.local_search.bfgs_b`.
        :param projection_callback_population: Same as `projection_callback`, but for
               the whole bacteria population. That is, in- and outputs must have
               the shape (n_bacteria, n_dims).
        :return: A `SingleSearchOutput` instance containing the optimization results.

        """
        x0_population_orig = x0_population.copy()
        x0_population, _ = projection_callback_population(x0_population)
        if not np.array_equal(x0_population, x0_population_orig):
            warnings.warn(
                message="Found initial conditions outside the defined search domain.",
                stacklevel=1,
            )

        n_bacteria, n_dims = x0_population.shape
        if n_bacteria < self.n_best_selection:
            err_msg = (
                "`n_best_selection` must not be larger than the number of bacteria."
            )
            raise ValueError(err_msg)

        global_search_result = run_and_tumble(
            f,
            x0_population,
            projection_callback_population,
            self.global_search_config,
        )
        # For deciding whether the overall search was successful, we ignore whether
        # the initial global search stage was successful (which is the case if the
        # bacteria distribution has reached a stationary state) and instead just
        # check whether any of the subsequent local searches were successful. This is
        # because the global search stage is not expected to find a minimum with high
        # accuracy, but rather to just explore the search space and provide good
        # starting points for the local searches.
        x_best_gs, f_best_gs, nfev_gs, nit_gs, _success_gs, trace_gs = (
            global_search_result.to_tuple()
        )
        if trace_gs is None:
            err_msg = "Global search did not return any trace. This should not happen."
            raise RuntimeError(err_msg)

        logger.debug(
            "=========================================================================="
        )
        logger.info(
            "Best result after run-and-tumble stage is x = %s, f(x) = %f. "
            "Starting local, gradient-based optimization for the %d best bacteria.",
            str(x_best_gs[np.argmin(f_best_gs)]),
            np.min(f_best_gs),
            self.n_best_selection,
        )

        sort_idx = f_best_gs.argsort()
        x_best_selection = x_best_gs[sort_idx[: self.n_best_selection]]
        x_best_ls = np.empty(x_best_selection.shape)
        f_min_ls = np.empty(self.n_best_selection)
        nfev_ls = 0
        nit_ls = 0
        success_ls = np.empty(self.n_best_selection)
        trace_ls = np.empty((self.local_search_config.niter, n_bacteria, n_dims))
        # Initialize with last point of global search trace
        trace_ls[:] = trace_gs[-1]
        nit_ls_arr = np.empty(self.n_best_selection)
        visited_points = trace_gs.reshape(-1, n_dims)

        for n, x_start in enumerate(x_best_selection):
            logger.debug("Performing gradient descent for bacterium %d.", n)

            hessian, nfev_hessian_calculation = _calculate_hessian_matrix(
                f,
                x_start,
                visited_points,
            )
            nfev_gs += nfev_hessian_calculation

            local_search_result = bfgs_b(
                f,
                x_start,
                projection_callback,
                hessian,
                self.local_search_config,
            )
            x_best_ls[n] = local_search_result.x_best
            f_min_ls[n] = local_search_result.f_best
            nfev_ls += local_search_result.nfev
            nit_ls += local_search_result.nit
            nit_ls_arr[n] = local_search_result.nit
            success_ls[n] = local_search_result.success
            local_trace = local_search_result.trace
            if local_trace is None:
                err_msg = (
                    "Local search did not return any trace. This should not happen."
                )
                raise RuntimeError(err_msg)
            trace_ls[:, sort_idx[n], :] = pad_trace(
                local_trace, self.local_search_config.niter
            )

        return SingleSearchOutput(
            x_best=x_best_ls[np.argmin(f_min_ls)],
            f_best=np.min(f_min_ls).astype(float),
            nfev=nfev_gs + nfev_ls,
            nit=nit_gs + nit_ls,
            success=success_ls.any().astype(bool),
            trace=np.concatenate(
                (trace_gs, trace_ls[: np.max(nit_ls_arr).astype(int)])
            ),
        )


def optimize(
    f: ObjectiveFunctionType,
    x0: ArrayLike | None = None,
    bounds: ArrayLike | ProjectionCallbackType | None = None,
    config: OptimizationConfig | None = None,
) -> OptimizeResult:
    """Optimizer entrypoint function.

    Metaheuristic global optimization algorithm combining a bacterial run-and-tumble
    chemotactic search with a local, gradient-based search around the best minimum
    candidate points.
    The algorithm's goal is to find
                                        min f(x), x ∈ Ω,
    where f: Ω ⊂ ℝ^n → ℝ.
    Since the chemotactic search becomes more and more inefficient with increasing
    problem dimensionality, Sequential Random Embeddings are used to solve the
    optimization problem once its dimensionality exceeds a given threshold.

    :param f: Objective function. Must accept its argument `x` as numpy array.
    :param x0: Optional initial conditions object. Must have the shape
           (n_bacteria, n_dims) or (n_dims,). If `x0` is None, initial conditions are
           sampled randomly or uniformly-spaced from Ω, depending on the parameter
           `init`. Note that this is only supported if Ω is a rectangular box, i.e., if
           no or non-rectangular bounds are imposed, `x0` must not be None.
    :param bounds: Defines the bounded domain Ω. If provided, i.e., if not None, must be
           one of the following:
           - Bounds projection callback, as defined in description of parameter
             `projection_callback` in :func:`search.local_search.bfgs_b`.
           - Rectangular box constraints. For each component `x_i` of x,
             `bounds[i, 0] <= x_i <= bounds[i, 1]` must hold, that is, `bounds` must
             have shape (n_dims, 2).
    :param config: Optimizer configuration.
    :return: An `OptimizeResult` instance containing the best found minimum of `f`.

    """
    if config is None:
        config = OptimizationConfig()

    min_allowed_n_reduced_dims = 2
    if config.embedding.n_reduced_dims < min_allowed_n_reduced_dims:
        err_msg = (
            f"`n_reduced_dims` must not be less than {min_allowed_n_reduced_dims}."
        )
        raise ValueError(err_msg)

    (
        n_bacteria,
        n_dims,
        projection_callback,
        projection_callback_population,
        x0_population,
        domain_scale,
    ) = _set_up_initial_and_boundary_conditions(x0, bounds, config)

    if config.n_best_selection > n_bacteria:
        err_msg = "`n_best_selection` must not be larger than `n_bacteria`."
        raise ValueError(err_msg)

    _auto_scale_parameters(config, domain_scale)

    optimizer = Optimizer(
        config.global_search, config.n_best_selection, config.local_search
    )

    if n_dims > config.max_dims:
        logger.info(
            "Using Sequential Random Embeddings in %d + 1 dimensions.",
            config.embedding.n_reduced_dims,
        )
        result = sequential_random_embeddings(
            f,
            x0_population,
            projection_callback,
            optimizer,
            config.embedding,
        )
        return result.to_optimize_result()

    result = optimizer.run(
        f, x0_population, projection_callback, projection_callback_population
    )
    return result.to_optimize_result()


def _set_up_initial_and_boundary_conditions(  # noqa: C901
    x0: ArrayLike | None,
    bounds: ArrayLike | ProjectionCallbackType | None,
    config: OptimizationConfig,
) -> tuple[
    int, int, ProjectionCallbackType, ProjectionCallbackType, np.ndarray, float | None
]:
    rng = np.random.default_rng(config.seed)
    n_reduced_dims_eff = config.embedding.n_reduced_dims + 1
    domain_scale = None

    if bounds is None or callable(bounds):
        if x0 is None:
            err_msg = (
                "If no box constraints are provided for `bounds`, "
                "`x0` must not be None."
            )
            raise ValueError(err_msg)
        x0_population = prepare_x0(
            x0,
            config.n_bacteria_per_dim,
            config.max_dims,
            n_reduced_dims_eff,
        )
        n_bacteria, n_dims = x0_population.shape

        if bounds is None:
            bound_lower, bound_upper = prepare_bounds(bounds, n_dims)

            def projection_callback(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                x = np.clip(x, bound_lower, bound_upper)
                bounds_hit = np.where(
                    ((x == bound_lower) | (x == bound_upper)), True, False
                )
                return x, bounds_hit

            def projection_callback_population(
                x: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray]:
                return projection_callback(x)

        else:

            def projection_callback(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                return bounds(x)

            def projection_callback_population(
                x: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray]:
                out = np.array([projection_callback(x_single) for x_single in x])
                return out[:, 0], out[:, 1]

    elif isinstance(bounds, (list, tuple, np.ndarray)):
        if x0 is not None:
            x0_population = prepare_x0(
                x0,
                config.n_bacteria_per_dim,
                config.max_dims,
                n_reduced_dims_eff,
            )
            n_bacteria, n_dims = x0_population.shape
            bound_lower, bound_upper = prepare_bounds(bounds, n_dims)
        else:
            n_dims = bounds.shape[0] if isinstance(bounds, np.ndarray) else len(bounds)
            bound_lower, bound_upper = prepare_bounds(bounds, n_dims)
            n_bacteria = (
                config.n_bacteria_per_dim**n_dims
                if n_dims <= config.max_dims
                else config.n_bacteria_per_dim**n_reduced_dims_eff
            )
            if config.init == "uniform" and n_dims > config.max_dims:
                config.init = "random"
                logger.warning(
                    "The option `init='uniform'` is only available for problems with "
                    "dimensionality less than or equal to `max_dims`, which was set to "
                    "%d. Since the current problem has dimensionality %d, `init` was "
                    "automatically set to 'random'.",
                    config.max_dims,
                    n_dims,
                )
            if config.init == "random":
                x0_population = rng.uniform(
                    bound_lower, bound_upper, size=(n_bacteria, n_dims)
                )
            elif config.init == "uniform":
                init_points = [
                    np.linspace(
                        bound_lower[i], bound_upper[i], config.n_bacteria_per_dim
                    )
                    for i in range(n_dims)
                ]
                x0_population = (
                    np.array(np.meshgrid(*init_points)).reshape(n_dims, -1).T
                )
            else:
                err_msg = "`init` must either be 'random' or 'uniform'."
                raise ValueError(err_msg)

        def projection_callback(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x = np.clip(x, bound_lower, bound_upper)
            bounds_hit = np.where(
                ((x == bound_lower) | (x == bound_upper)), True, False
            )
            return x, bounds_hit

        def projection_callback_population(
            x: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            return projection_callback(x)

        domain_range = bound_upper - bound_lower
        domain_scale = np.max(np.where(np.isinf(domain_range), 0, domain_range))

    else:
        err_msg = (
            "`bounds` must either be None, an array or corresponding nested list of "
            "shape (n_dims, 2), or a custom callback function."
        )
        raise ValueError(err_msg)

    return (
        n_bacteria,
        n_dims,
        projection_callback,
        projection_callback_population,
        x0_population,
        domain_scale,
    )


def _auto_scale_parameters(
    config: OptimizationConfig, domain_scale: float | None
) -> None:
    if domain_scale is not None and domain_scale > 0:
        if config.global_search.stepsize_start is None:
            config.global_search.stepsize_start = 0.1 * domain_scale
        if config.global_search.attraction_sigma is None:
            config.global_search.attraction_sigma = domain_scale
        if config.local_search.a is None:
            config.local_search.a = 1e-2 * domain_scale
    else:
        if config.global_search.stepsize_start is None:
            config.global_search.stepsize_start = 0.1
            warnings.warn(
                message="`stepsize_start` was not provided. As auto-scaling of "
                "`stepsize_start` is not possible for the given problem, it has been "
                "set to 0.1 by default. Please consider tuning this parameter to your "
                "specific problem.",
                stacklevel=1,
            )
        if config.global_search.attraction_sigma is None:
            config.global_search.attraction_sigma = 1
            warnings.warn(
                message="`attraction_sigma` was not provided. As auto-scaling of "
                "`attraction_sigma` is not possible for the given problem, it has been "
                "set to 1 by default. Please consider tuning this parameter to your "
                "specific problem.",
                stacklevel=1,
            )
        if config.local_search.a is None:
            config.local_search.a = 1e-2
            warnings.warn(
                message="The initial local search step size `a` was not provided. As "
                "auto-scaling of `a` is not possible for the given problem, it has "
                "been set to 1e-2 by default. Please consider tuning this parameter to "
                "your specific problem.",
                stacklevel=1,
            )


def _calculate_hessian_matrix(
    f: ObjectiveFunctionType,
    x: np.ndarray,
    visited_points: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Calculate Hessian matrix via quadratic function approximation around `x`.

    :param f: Objective function. Must accept its argument `x` as numpy array.
    :param x: Point around which to calculate the Hessian matrix.
    :param visited_points: Points visited so far by the global search routine.
    :return: Estimated Hessian matrix at `x` and number of objective function
             evaluations taken.

    """
    # Step 1: Calculate quadratic function approximation around `x`.
    n_dims = x.shape[0]
    num_sampling_points = 2 * int(special.binom(n_dims + 2, 2))
    sampling_points = visited_points[
        spatial.cKDTree(visited_points).query(x, num_sampling_points)[1]
    ]
    func_values = np.array([f(point) for point in sampling_points])
    polynomial_powers = list(
        itertools.filterfalse(
            lambda prod: sum(list(prod)) > 2,  # noqa: PLR2004
            itertools.product((0, 1, 2), repeat=n_dims),
        )
    )
    sampling_matrix = np.stack(
        [np.prod(sampling_points**d, axis=1) for d in polynomial_powers],
        axis=-1,
    )
    coeffs = np.linalg.lstsq(sampling_matrix, func_values, 2)[0]

    # Step 2: Calculate Hessian matrix from the quadratic approximation.
    hessian = np.ones((n_dims, n_dims))
    square_powers = list(
        itertools.filterfalse(
            lambda zipped_item: sum(list(zipped_item[0])) != 2,  # noqa: PLR2004
            zip(polynomial_powers, coeffs, strict=True),
        )
    )
    for square_power, coeff in square_powers:
        idcs_to_consider = np.argwhere(np.array(square_power) != 0)
        if len(idcs_to_consider) == 1:  # Diagonal
            hessian[idcs_to_consider[0], idcs_to_consider[0]] = 0.5 * coeff
        elif len(idcs_to_consider) == 2:  # Mixed derivatives  # noqa: PLR2004
            hessian[idcs_to_consider[0], idcs_to_consider[1]] = coeff
            hessian[idcs_to_consider[1], idcs_to_consider[0]] = coeff
        else:
            err_msg = (
                "Polynomial function approximation seems to be of higher order "
                "than two. This shouldn't happen."
            )
            raise RuntimeError(err_msg)
    return hessian, num_sampling_points
