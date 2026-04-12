import numpy as np
import pytest

from rt_opt.config.global_search import RunAndTumbleConfig
from rt_opt.search.global_search import run_and_tumble


class ObjectiveFunction:
    def __init__(self) -> None:
        self.call_counter = 0

    def __call__(self, x: np.ndarray) -> float:
        self.call_counter += 1
        return np.square(x).sum()


@pytest.mark.filterwarnings("ignore:`attraction_sigma` was not provided.")
@pytest.mark.filterwarnings("ignore:`stepsize_start` was not provided.")
@pytest.mark.parametrize("bounds", [(-3.0, 3.0), (-np.inf, np.inf)])
@pytest.mark.parametrize("stepsize_start", [0.1, None])
@pytest.mark.parametrize("stationarity_window", [20, 1000])
@pytest.mark.parametrize("attraction", [True, False])
@pytest.mark.parametrize("attraction_sigma", [1, None])
@pytest.mark.parametrize("bounds_reflection", [True, False])
def test_run_and_tumble(
    bounds: tuple[float, float],
    stepsize_start: float | None,
    stationarity_window: int,
    attraction: bool,
    attraction_sigma: float | None,
    bounds_reflection: bool,
) -> None:
    config = RunAndTumbleConfig(
        niter=400,
        stepsize_start=stepsize_start,
        stepsize_decay_fac=1e-3,
        base_tumble_rate=0.1,
        stationarity_window=stationarity_window,
        eps_stat=5e-2,
        stationarity_r_value_threshold=0.9,
        attraction=attraction,
        attraction_window=10,
        attraction_sigma=attraction_sigma,
        attraction_strength=0.5,
        bounds_reflection=bounds_reflection,
        seed=42,
    )
    bound_lower, bound_upper = bounds

    def projection_callback(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.clip(x, bound_lower, bound_upper)
        bounds_hit = np.where(((x == bound_lower) | (x == bound_upper)), True, False)
        return x, bounds_hit

    x0_population = np.array([[1.0, 2.0], [3.0, 4.0]])
    objective_function = ObjectiveFunction()
    if stationarity_window >= config.niter:
        with pytest.raises(
            ValueError, match=r"`niter` must be larger than `stationarity_window`."
        ):
            run_and_tumble(
                objective_function, x0_population, projection_callback, config
            )
    else:
        if stepsize_start is None:
            with pytest.warns(
                UserWarning,
                match=r"`stepsize_start` was not provided.",
            ):
                result = run_and_tumble(
                    objective_function, x0_population, projection_callback, config
                )
        elif attraction_sigma is None:
            with pytest.warns(
                UserWarning,
                match=r"`attraction_sigma` was not provided.",
            ):
                result = run_and_tumble(
                    objective_function, x0_population, projection_callback, config
                )
        else:
            result = run_and_tumble(
                objective_function, x0_population, projection_callback, config
            )

        n_bacteria = x0_population.shape[0]

        assert isinstance(result.x_best, np.ndarray)
        assert isinstance(result.f_best, np.ndarray)
        assert isinstance(result.trace, np.ndarray)

        assert result.x_best.shape == x0_population.shape
        assert result.f_best.shape == (n_bacteria,)
        assert result.nfev == objective_function.call_counter
        assert all(
            f_best == pytest.approx(objective_function(x_best), abs=np.finfo(float).eps)
            for f_best, x_best in zip(result.f_best, result.x_best, strict=True)
        )
        assert (
            result.success is True or result.nit == config.niter
        )  # Algorithm my or may not converge
        assert result.trace.shape == (result.nit + 1, *x0_population.shape)
        assert np.array_equal(result.trace[0], x0_population)
