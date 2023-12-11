"""Shared testing fixtures."""

import pytest

from freebus.experiments import Experiment, Routes
from freebus.randomvar import Fixed, FixedAlternating


@pytest.fixture
def deterministic_experiment():
    """Provides a simple deterministic experiment object."""
    return Experiment(
        routes=Routes(
            [2],
            distance=[[1, 0]],
            traffic=Fixed(1),
            demand_loading=FixedAlternating([[[1, 0], [0, 0]]]),
            demand_unloading=Fixed([[0, 1]]),),
        time_loading=Fixed(1),
        time_unloading=Fixed(1),
        schedule=[[10]],
        headers=['waiting-time', 'loading-time', 'moving-time',
                 'holding-time', 'total-passengers']
    )
