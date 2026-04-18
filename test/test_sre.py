import numpy as np
import pytest
from common.objective_functions import SphereObjectiveFunction

from rt_opt.config.global_search import RunAndTumbleConfig
from rt_opt.config.local_search import BFGSBConfig
from rt_opt.config.optimizer import SequentialRandomEmbeddingsConfig
from rt_opt.dimension_reduction.sre import sequential_random_embeddings
from rt_opt.optimizer import Optimizer
from rt_opt.utils.types import ProjectionCallbackType


def projection_callback_unbounded(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x, np.zeros_like(x, dtype=bool)


def projection_callback_box(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bound_lower = 1.0 * np.ones(x.shape)
    bound_upper = 7.0 * np.ones(x.shape)
    x = np.clip(x, bound_lower, bound_upper)
    bounds_hit = np.where(((x == bound_lower) | (x == bound_upper)), True, False)

    return x, bounds_hit


@pytest.mark.parametrize(
    ("projection_callback", "expected_f_best_per_dim"),
    [(projection_callback_unbounded, 0), (projection_callback_box, 1)],
)
@pytest.mark.parametrize(
    "x0", [np.array([[1.0, 2.0, 3.0, 4.0]]), 4.0 * np.ones((20, 1))]
)
def test_sequential_random_embeddings(
    projection_callback: ProjectionCallbackType,
    expected_f_best_per_dim: float,
    x0: np.ndarray,
) -> None:
    optimizer = Optimizer(
        global_search_config=RunAndTumbleConfig(
            seed=42, stepsize_start=0.1, attraction_sigma=1
        ),
        n_best_selection=1,
        local_search_config=BFGSBConfig(a=1),
    )
    config = SequentialRandomEmbeddingsConfig(n_embeddings=3, n_reduced_dims=2, seed=42)
    objective_function = SphereObjectiveFunction()
    result = sequential_random_embeddings(
        objective_function, x0, projection_callback, optimizer, config
    )

    expected_n_dims = x0.shape[1]
    assert isinstance(result.x_best, np.ndarray)
    assert isinstance(result.f_best, float)
    assert result.success
    assert result.nfev == objective_function.call_counter
    assert result.x_best.shape == (expected_n_dims,)
    assert result.f_best == pytest.approx(
        objective_function(result.x_best), abs=np.finfo(float).eps
    )
    assert result.f_best == pytest.approx(
        expected_f_best_per_dim * expected_n_dims, abs=1e-6
    )
    assert result.trace is None
