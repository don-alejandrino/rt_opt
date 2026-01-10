from typing import Literal

import numpy as np
import pytest
from numpy._typing import ArrayLike

from rt_opt.config.local_search import AdamSPSAConfig, BFGSBConfig, LineSearchConfig
from rt_opt.search.local_search import adam_spsa, bfgs_b, two_way_linesearch


class ObjectiveFunction:
    def __init__(self) -> None:
        self.call_counter = 0

    def __call__(self, x: np.ndarray) -> float:
        self.call_counter += 1
        return (x**4).sum()

    @staticmethod
    def grad(x: np.ndarray) -> np.ndarray:
        return 4.0 * x**3


@pytest.mark.parametrize("bounds", [(-3.0, 3.0), (-np.inf, np.inf)])
@pytest.mark.parametrize("a", [0.1, None])
@pytest.mark.parametrize(
    "hessian_start", [np.array([[48, 0], [0, 48]], dtype=float), None]
)
@pytest.mark.parametrize("niter", [5, 100])
def test_bfgs_b(
    bounds: ArrayLike,
    a: float | None,
    hessian_start: np.ndarray | None,
    niter: Literal[5, 100],
) -> None:
    config = BFGSBConfig(
        c=1e-6,
        a=a,
        niter=niter,
        eps_abs=1e-9,
        eps_rel=1e-6,
        linesearch=LineSearchConfig(),
    )
    bound_lower, bound_upper = bounds

    def projection_callback(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.clip(x, bound_lower, bound_upper)
        bounds_hit = np.where(((x == bound_lower) | (x == bound_upper)), True, False)
        return x, bounds_hit

    x0 = np.array([2.0, 2.0])
    objective_function = ObjectiveFunction()
    if a is None:
        with pytest.warns(
            UserWarning,
            match=r"Initial search step size `a` was not provided.",
        ):
            result = bfgs_b(
                objective_function, x0, projection_callback, hessian_start, config
            )
    else:
        result = bfgs_b(
            objective_function, x0, projection_callback, hessian_start, config
        )

    assert isinstance(result.x_best, np.ndarray)
    assert result.x_best.shape == x0.shape
    assert isinstance(result.f_best, float)
    assert result.nfev == objective_function.call_counter
    assert isinstance(result.trace, np.ndarray)
    assert result.trace.shape == (result.nit, *x0.shape)
    if niter == 100:
        # For this simple objective function, we expect the algorithm to always converge
        # within less than 100 iterations
        assert result.success is True
        assert result.nit < config.niter
        assert result.f_best == pytest.approx(0, abs=1e-6)
    elif niter == 5:
        # For only 5 iterations, we expect no convergence
        assert result.success is False
        assert result.nit == config.niter
    else:
        err_msg = "Unexpected number of iterations."
        raise ValueError(err_msg)


@pytest.mark.parametrize("bounds", [(-3.0, 3.0), (-np.inf, np.inf)])
@pytest.mark.parametrize(
    ("a", "expected_direction"), [(10.0, "decrease"), (1e-4, "increase")]
)
@pytest.mark.parametrize(("niter", "expect_success"), [(20, True), (2, False)])
def test_two_way_linesearch(
    bounds: ArrayLike,
    a: float,
    niter: int,
    expected_direction: Literal["decrease", "increase"],
    expect_success: bool,
) -> None:
    config = LineSearchConfig(niter=niter)
    bound_lower, bound_upper = bounds

    def projection_callback(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.clip(x, bound_lower, bound_upper)
        bounds_hit = np.where(((x == bound_lower) | (x == bound_upper)), True, False)
        return x, bounds_hit

    x0 = np.array([2.0, 2.0])
    objective_function = ObjectiveFunction()
    f_old = objective_function(x0)
    grad = objective_function.grad(x0)
    d = -grad  # Search in the direction of the negative gradient

    # Reset the counter for the actual test
    objective_function.call_counter = 0
    result = two_way_linesearch(
        objective_function, x0, grad, d, a, f_old, projection_callback, config
    )

    assert isinstance(result.x, np.ndarray)
    assert result.x.shape == x0.shape
    assert isinstance(result.f, float)
    assert isinstance(result.a, float)
    assert result.nfev == objective_function.call_counter
    assert isinstance(result.nit, int)
    assert result.success is expect_success

    if expect_success:
        assert result.f < f_old
        assert result.nit < niter
        if expected_direction == "decrease":
            assert result.a < a
        elif expected_direction == "increase":
            assert result.a > a
        else:
            err_msg = "Unexpected value for `expected_direction`."
            raise ValueError(err_msg)
    else:
        assert result.nit == niter


@pytest.mark.parametrize("bounds", [(-3.0, 3.0), (-np.inf, np.inf)])
@pytest.mark.parametrize("x0", [(2.0, 2.0), (-4.0, 2.0)])
@pytest.mark.parametrize("niter", [10, 500])
def test_adam_spsa(
    x0: ArrayLike,
    bounds: ArrayLike,
    niter: Literal[5, 100],
) -> None:
    config = AdamSPSAConfig(seed=42, niter=niter)
    x0 = np.array(x0)
    objective_function = ObjectiveFunction()

    def projection_callback(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bound_lower, bound_upper = bounds
        x = np.clip(x, bound_lower, bound_upper)
        bounds_hit = np.where(((x == bound_lower) | (x == bound_upper)), True, False)
        return x, bounds_hit

    # Reset counter for the actual test
    objective_function.call_counter = 0
    if not np.array_equal(projection_callback(x0)[0], x0):
        with pytest.raises(
            ValueError,
            match=(
                r"`x0` is outside the bounded domain defined by `projection_callback`."
            ),
        ):
            adam_spsa(objective_function, x0, projection_callback, config)
    else:
        result = adam_spsa(objective_function, x0, projection_callback, config)

        assert isinstance(result.x_best, np.ndarray)
        assert result.x_best.shape == x0.shape
        assert isinstance(result.f_best, float)
        assert result.nfev == objective_function.call_counter
        assert isinstance(result.trace, np.ndarray)
        assert result.trace.shape == (result.nit, *x0.shape)
        if niter == 500:
            # For this simple objective function, we expect the algorithm to always
            # converge within less than 500 iterations
            assert result.success is True
            assert result.nit < config.niter
            assert result.f_best == pytest.approx(0, abs=1e-9)
        elif niter == 10:
            # For only 10 iterations, we expect no convergence
            assert result.success is False
            assert result.nit == config.niter
        else:
            err_msg = "Unexpected number of iterations."
            raise ValueError(err_msg)
