"""Test estimation and simulation mechanisms."""

from pytest import approx

import freebus as fb

# def test_estimate(monkeypatch):
#     """estimate(experiment, n) should return the mean result, with
#     experiment parameters, as performed on n trials."""
#     def simulate_mock(*args, **kwargs):
#         return next(mock_trials)
#     mock_trials = (i for i in range(3))
#     monkeypatch.setattr(fb.main, 'simulate', simulate_mock)
#     result = fb.main.estimate(None, 3)
#     assert result == approx(1)
