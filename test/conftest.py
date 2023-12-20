"""Shared testing fixtures."""

import pytest

from freebus.experiments import Experiment, Routes, TrafficModel
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


@pytest.fixture
def ReturnFrom():
    """Return a callable type that sequentially returns elements from an
    iterable."""
    class ReturnFrom_:
        """A function that returns elements from a list, in order."""
        def __init__(self, returns: list):
            self.returns = iter(returns)

        def __call__(self, *_, **__):
            try:
                return next(self.returns)
            except StopIteration:
                return None
    return ReturnFrom_
