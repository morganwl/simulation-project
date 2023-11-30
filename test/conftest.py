"""Shared testing fixtures."""

import pytest

@pytest.fixture
def deterministic_experiment():
    """Provides a simple deterministic experiment object."""
    class Experiment:
    # pylint: disable=missing-docstring, unused-argument
        routes = [2]
        distance = [[1, 0]]
        schedule = [[10]]
        headers = ['loading-time', 'moving-time', 'holding-time', 'total-passengers']
        def traffic(self, r, s, t):
            return 1

        def demand_loading(self, r, s, t, d):
            if s == 0 and d > 1:
                return 1
            return 0

        def demand_unloading(self, r, s, t):
            return [0,1][s]

        def time_loading(self, p):
            return 1

        def time_unloading(self, p):
            return 1

        def __repr__(self):
            return (f'{type(self).__name__}('
            f'{self.routes}, {self.distance}, {self.schedule}, {self.headers})')
    return Experiment()
