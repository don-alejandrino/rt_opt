from collections.abc import Sequence

import numpy as np
import pytest

from rt_opt.config.global_search import RunAndTumbleConfig
from rt_opt.config.local_search import BFGSBConfig
from rt_opt.optimizer import Optimizer


class ObjectiveFunction:
    def __init__(self) -> None:
        self.call_counter = 0

    def __call__(self, x: np.ndarray) -> float:
        self.call_counter += 1
        return np.square(x).sum()


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

    objective_function = ObjectiveFunction()
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
    else:
        if (projection_callback_population(x0_population)[0] != x0_population).any():
            with pytest.warns(
                UserWarning,
                match=r"Found initial conditions outside the defined search domain.",
            ):
                result = optimizer.run(**optimization_args)
        else:
            result = optimizer.run(**optimization_args)

        assert isinstance(result.x_best, np.ndarray)
        assert isinstance(result.f_best, float)
        assert isinstance(result.trace, np.ndarray)

        assert result.success
        assert result.nfev == objective_function.call_counter
        assert result.nit < global_search_config.niter + local_search_config.niter
        assert result.x_best.shape == (x0_population.shape[1],)
        trace_length = result.trace.shape[0]
        assert trace_length <= result.nit + 1  # +1 for initial population
        assert result.trace.shape[1:] == x0_population.shape
        assert np.array_equal(
            result.trace[0], projection_callback_population(x0_population)[0]
        )
        assert result.f_best == pytest.approx(
            objective_function(result.x_best), abs=np.finfo(float).eps
        )
        assert result.f_best == pytest.approx(0, abs=1e-6)


# TODO(Alex): Test optimize entrypoint
