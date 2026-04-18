from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pytest
from common.objective_functions import SphereObjectiveFunction
from numpy import ndarray
from scipy.optimize import OptimizeResult

import rt_opt.optimizer as optimizer_module
from rt_opt.config.global_search import RunAndTumbleConfig
from rt_opt.config.local_search import BFGSBConfig
from rt_opt.config.optimizer import (
    OptimizationConfig,
    SequentialRandomEmbeddingsConfig,
)
from rt_opt.dimension_reduction.sre import sequential_random_embeddings
from rt_opt.optimizer import Optimizer
from rt_opt.search.search_output import SingleSearchOutput
from rt_opt.utils.io import prepare_x0
from rt_opt.utils.types import ProjectionCallbackType

BoundsType = np.ndarray | ProjectionCallbackType | Sequence[Sequence[float]] | None


class SequentialRandomEmbeddingsWrapper:
    def __init__(self) -> None:
        self.call_args = None

    def __call__(
        self,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> SingleSearchOutput:
        self.call_args = args
        self.call_kwargs = kwargs

        return sequential_random_embeddings(*args, **kwargs)


def box_projection_callback(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bound_lower = -5.0 * np.ones(x.shape)
    bound_upper = 5.0 * np.ones(x.shape)
    x = np.clip(x, bound_lower, bound_upper)
    bounds_hit = np.where(((x == bound_lower) | (x == bound_upper)), True, False)

    return x, bounds_hit


def make_optimization_config(
    init: Literal["random", "uniform"] = "uniform",
    n_best_selection: int = 2,
    max_dims: int = 3,
    n_reduced_dims: int = 2,
    n_bacteria_per_dim: int = 2,
) -> OptimizationConfig:
    return OptimizationConfig(
        init=init,
        n_bacteria_per_dim=n_bacteria_per_dim,
        n_best_selection=n_best_selection,
        max_dims=max_dims,
        global_search=RunAndTumbleConfig(
            stepsize_start=0.1,
            attraction_sigma=1,
            seed=123,
        ),
        local_search=BFGSBConfig(a=0.1),
        embedding=SequentialRandomEmbeddingsConfig(
            n_embeddings=3,
            n_reduced_dims=n_reduced_dims,
            seed=456,
        ),
        seed=789,
    )


def check_optimization_result(
    objective_function: SphereObjectiveFunction,
    result: OptimizeResult | SingleSearchOutput,
    global_search_niter_max: int,
    local_search_niter_max: int,
    expected_population_shape: tuple[int, ...],
    bounds: BoundsType,
    sre_expected: bool = False,
    expected_f_min: float = 0,
) -> None:
    if isinstance(result, OptimizeResult):
        x_best: np.ndarray = result.x
        f_best: float = result.fun
        trace: np.ndarray | None = getattr(result, "trace", None)
    elif isinstance(result, SingleSearchOutput):
        x_best = result.x_best
        f_best = result.f_best
        trace = result.trace
    else:
        err_msg = f"Unexpected result type: {type(result)}."
        raise TypeError(err_msg)
    nfev = result.nfev
    nit = result.nit
    success = result.success

    assert isinstance(x_best, np.ndarray)
    assert isinstance(f_best, float)
    assert success
    assert nfev == objective_function.call_counter
    if not sre_expected:
        assert 0 < nit < global_search_niter_max + local_search_niter_max
    assert x_best.shape == (expected_population_shape[1],)
    assert f_best == pytest.approx(objective_function(x_best), abs=np.finfo(float).eps)
    assert f_best == pytest.approx(expected_f_min, abs=1e-6)

    if sre_expected:
        assert trace is None
    else:
        assert isinstance(trace, np.ndarray)
        assert trace.shape[0] <= nit + 1  # +1 for initial population
        assert trace.shape[1:] == expected_population_shape
        if isinstance(bounds, (np.ndarray, Sequence)):
            bound_lower = np.array(bounds)[:, 0]
            bound_upper = np.array(bounds)[:, 1]
            assert np.all(trace >= bound_lower)
            assert np.all(trace <= bound_upper)
        elif isinstance(bounds, Callable):
            clipped_trace = np.apply_along_axis(lambda x: bounds(x)[0], -1, trace)
            assert np.all(trace == clipped_trace)


def get_expected_population_shape(
    x0: ndarray | None,
    bounds: BoundsType,
    config: OptimizationConfig,
) -> tuple[int, int]:
    if x0 is not None:
        if x0.ndim == 1:
            ndims = len(x0)
            expected_population_shape = (config.n_bacteria_per_dim**ndims, ndims)
        else:
            expected_population_shape = x0.shape
            assert len(expected_population_shape) == 2
    else:
        assert bounds is not None
        assert not isinstance(bounds, Callable)
        ndims = len(bounds)
        expected_population_shape = (config.n_bacteria_per_dim**ndims, ndims)

    return expected_population_shape


@pytest.mark.filterwarnings("ignore:Found initial conditions outside")
@pytest.mark.parametrize("bounds", [(-5.0, 5.0), (-np.inf, np.inf)])
@pytest.mark.parametrize(
    "x0_population",
    [
        np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 4.0], [3.0, 2.0]]),
        np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 4.0], [3.0, 6.0]]),
    ],
)
@pytest.mark.parametrize("n_best_selection", [1, 2, 5])
def test_optimizer_run(
    bounds: Sequence[Sequence[float]], x0_population: np.ndarray, n_best_selection: int
) -> None:
    bound_lower, bound_upper = bounds

    def projection_callback(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.clip(x, bound_lower, bound_upper)
        bounds_hit = np.where(((x == bound_lower) | (x == bound_upper)), True, False)
        return x, bounds_hit

    def projection_callback_population(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return projection_callback(x)

    global_search_config = RunAndTumbleConfig(
        stepsize_start=0.1,
        attraction_sigma=1,
        seed=123,
    )
    local_search_config = BFGSBConfig(a=0.1)
    optimizer = Optimizer(
        global_search_config=global_search_config,
        n_best_selection=n_best_selection,
        local_search_config=local_search_config,
    )

    objective_function = SphereObjectiveFunction()
    optimization_args = {
        "f": objective_function,
        "x0_population": x0_population,
        "projection_callback": projection_callback,
        "projection_callback_population": projection_callback_population,
    }
    if n_best_selection > x0_population.shape[0]:
        with pytest.raises(
            ValueError,
            match=r"`n_best_selection` must not be larger than the number of bacteria.",
        ):
            optimizer.run(**optimization_args)
        return

    if (projection_callback_population(x0_population)[0] != x0_population).any():
        with pytest.warns(
            UserWarning,
            match=r"Found initial conditions outside the defined search domain.",
        ):
            result = optimizer.run(**optimization_args)
    else:
        result = optimizer.run(**optimization_args)

    assert isinstance(result, SingleSearchOutput)
    trace = result.trace
    assert trace is not None
    assert np.array_equal(trace[0], projection_callback_population(x0_population)[0])
    check_optimization_result(
        objective_function=objective_function,
        result=result,
        global_search_niter_max=global_search_config.niter,
        local_search_niter_max=local_search_config.niter,
        expected_population_shape=x0_population.shape,
        bounds=projection_callback,
    )


@pytest.mark.filterwarnings("ignore:The number of bacteria given by `x0`")
@pytest.mark.parametrize(
    ("max_dims", "x0", "bounds", "sre_expected"),
    [
        (3, np.array([1.0, -2.0]), None, False),
        (3, None, np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]), False),
        (3, np.array([[1.0, -2.0, 3.0], [3.0, -4.0, 5.0]]), None, False),
        (3, np.array([1.0, -2.0, 3.0, 4.0]), None, True),
        (2, None, np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]), True),
    ],
)
def test_optimize_invokes_sre_when_necessary(
    monkeypatch: pytest.MonkeyPatch,
    max_dims: int,
    x0: np.ndarray | None,
    bounds: np.ndarray | None,
    sre_expected: bool,
) -> None:
    sre_wrapper = SequentialRandomEmbeddingsWrapper()
    monkeypatch.setattr(
        optimizer_module,
        "sequential_random_embeddings",
        sre_wrapper,
    )

    config = make_optimization_config(max_dims=max_dims)
    objective_function = SphereObjectiveFunction()
    result = optimizer_module.optimize(
        objective_function, x0=x0, bounds=bounds, config=config
    )

    expected_population_shape = get_expected_population_shape(x0, bounds, config)

    check_optimization_result(
        objective_function=objective_function,
        result=result,
        global_search_niter_max=config.global_search.niter,
        local_search_niter_max=config.local_search.niter,
        expected_population_shape=expected_population_shape,
        bounds=bounds,
        sre_expected=sre_expected,
    )
    if sre_expected:
        args = sre_wrapper.call_args
        assert args is not None
        assert args[0] is objective_function
        if x0 is not None:
            expected_x0_population = prepare_x0(
                x0,
                config.n_bacteria_per_dim,
                config.max_dims,
                config.embedding.n_reduced_dims + 1,
            )
            assert np.array_equal(args[1], expected_x0_population)
        else:
            assert args[1].shape == expected_population_shape
        assert isinstance(args[2], Callable)
        assert isinstance(args[3], Optimizer)
        assert args[3].global_search_config is config.global_search
        assert args[3].n_best_selection == config.n_best_selection
        assert args[3].local_search_config is config.local_search
        assert args[4] is config.embedding
    else:
        assert sre_wrapper.call_args is None


@pytest.mark.parametrize("n_reduced_dims", [0, 1])
def test_optimize_rejects_too_few_reduced_dims(n_reduced_dims: int) -> None:
    config = make_optimization_config(n_reduced_dims=n_reduced_dims)
    with pytest.raises(
        ValueError,
        match=r"`n_reduced_dims` must not be less than 2\.",
    ):
        optimizer_module.optimize(
            SphereObjectiveFunction(),
            x0=np.array([1.0, 2.0]),
            config=config,
        )


def test_optimize_validates_n_best_selection_against_population() -> None:
    n_bacteria = 4
    n_best_selection = 5
    config = make_optimization_config(n_best_selection=n_best_selection)
    x0 = np.ones((n_bacteria, 2))
    with pytest.raises(
        ValueError,
        match=r"`n_best_selection` must not be larger than `n_bacteria`\.",
    ):
        optimizer_module.optimize(
            SphereObjectiveFunction(),
            x0=x0,
            config=config,
        )


@pytest.mark.parametrize("bounds", [None, box_projection_callback])
def test_optimize_requires_x0_without_box_bounds(bounds: BoundsType) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"If no box constraints are provided for `bounds`, `x0` must not be None\."
        ),
    ):
        optimizer_module.optimize(
            SphereObjectiveFunction(),
            x0=None,
            bounds=bounds,
            config=make_optimization_config(),
        )


@pytest.mark.parametrize("init", ["random", "uniform"])
def test_optimize_end_to_end_samples_initial_population_from_box_bounds(
    init: Literal["random", "uniform"],
) -> None:
    objective_function = SphereObjectiveFunction()
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    config = make_optimization_config(init=init, n_bacteria_per_dim=3)
    result = optimizer_module.optimize(
        objective_function,
        x0=None,
        bounds=bounds,
        config=config,
    )

    check_optimization_result(
        objective_function=objective_function,
        result=result,
        global_search_niter_max=config.global_search.niter,
        local_search_niter_max=config.local_search.niter,
        expected_population_shape=(9, 2),
        bounds=bounds,
    )

    trace: np.ndarray = result.trace
    assert np.all(trace[0] >= bounds[:, 0])
    assert np.all(trace[0] <= bounds[:, 1])
    if init == "uniform":
        assert np.array_equal(np.unique(trace[0, :, 0]), np.array([-5.0, 0, 5.0]))
        assert np.array_equal(np.unique(trace[0, :, 1]), np.array([-5.0, 0, 5.0]))


@pytest.mark.parametrize(
    "x0",
    [
        None,
        np.array([[1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [1.0, 1.0]]),
        np.array([1.0, 2.0, 3.0]),
        0.75 * np.ones((8, 3)),
    ],
)
@pytest.mark.parametrize(
    "bounds",
    [
        None,
        box_projection_callback,
        np.array([[0.5, 5.0], [0.5, 5.0], [0.5, 5.0]]),
    ],
)
@pytest.mark.parametrize("max_dims", [2, 3])
def test_optimize_end_to_end(
    x0: np.ndarray | None,
    bounds: BoundsType,
    max_dims: int,
) -> None:
    objective_function = SphereObjectiveFunction()
    config = OptimizationConfig(
        n_bacteria_per_dim=2,
        max_dims=max_dims,
        global_search=RunAndTumbleConfig(
            niter=100, stepsize_start=0.1, attraction_sigma=0.1, seed=666
        ),
        embedding=SequentialRandomEmbeddingsConfig(seed=666),
        seed=666,
    )
    if x0 is None and (bounds is None or isinstance(bounds, Callable)):
        with pytest.raises(
            ValueError,
            match=(
                r"If no box constraints are provided for `bounds`, `x0` must not be "
                r"None\."
            ),
        ):
            optimizer_module.optimize(
                objective_function,
                x0=x0,
                bounds=bounds,
                config=config,
            )
        return

    ndims_implied_by_bounds = len(bounds) if isinstance(bounds, np.ndarray) else None
    if x0 is None:
        ndims_implied_by_x0 = None
    elif x0.ndim == 1:
        ndims_implied_by_x0 = len(x0)
    else:
        ndims_implied_by_x0 = x0.shape[1]
    if (
        ndims_implied_by_bounds is not None
        and ndims_implied_by_x0 is not None
        and ndims_implied_by_bounds != ndims_implied_by_x0
    ):
        with pytest.raises(ValueError, match=r"`bounds` has wrong shape"):
            optimizer_module.optimize(
                objective_function,
                x0=x0,
                bounds=bounds,
                config=config,
            )
        return

    if not isinstance(bounds, np.ndarray):
        with pytest.warns(
            UserWarning,
            match=r"The initial local search step size `a` was not provided",
        ):
            result = optimizer_module.optimize(
                objective_function,
                x0=x0,
                bounds=bounds,
                config=config,
            )
    else:
        result = optimizer_module.optimize(
            objective_function,
            x0=x0,
            bounds=bounds,
            config=config,
        )

    expected_population_shape = get_expected_population_shape(x0, bounds, config)
    ndims = expected_population_shape[1]
    check_optimization_result(
        objective_function=objective_function,
        result=result,
        global_search_niter_max=config.global_search.niter,
        local_search_niter_max=config.local_search.niter,
        expected_population_shape=expected_population_shape,
        bounds=bounds,
        sre_expected=ndims > max_dims,
        expected_f_min=0.5**2 * 3 if isinstance(bounds, np.ndarray) else 0,
    )
