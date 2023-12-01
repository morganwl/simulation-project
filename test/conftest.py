"""Shared testing fixtures."""

import pytest

from freebus.experiments import Experiment
from freebus.randomvar import Fixed, FixedAlternating

@pytest.fixture
def deterministic_experiment():
    """Provides a simple deterministic experiment object."""
    # class Experiment:
    # # pylint: disable=missing-docstring, unused-argument
    #     routes = [2]
    #     distance = [[1, 0]]
    #     schedule = [[10]]
    #     headers = ['loading-time', 'moving-time', 'holding-time', 'total-passengers']
    #     traffic = Fixed(1)
    #     demand_loading = FixedAlternating([[[1,0], [0,0]]])
    #     demand_unloading = Fixed([[0, 1]])
    #     time_loading = Fixed(1)
    #     time_unloading = Fixed(1)

    #     def __repr__(self):
    #         return (f'{type(self).__name__}('
    #         f'{self.routes}, {self.distance}, {self.schedule}, {self.headers}, '
    #         f'traffic={self.traffic}, demand_loading={self.demand_loading}, '
    #         f'demand_unloading={self.demand_unloading}, time_loading={self.time_loading})')
    return Experiment(
            routes=[2],
            distance=[[1,0]],
            schedule=[[10]],
            traffic=Fixed(1),
            demand_loading=FixedAlternating([[[1,0], [0,0]]]),
            demand_unloading=Fixed([[0, 1]]),
            time_loading=Fixed(1),
            time_unloading=Fixed(1),
            headers=['loading-time', 'moving-time', 'holding-time', 'total-passengers']
            )
