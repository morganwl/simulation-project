"""Tests generation of measurements from a single trial."""

import numpy as np
import pytest
from pytest import fixture, approx

import freebus as fb
from freebus.types import Event


@fixture
def traffic_instances():
    """Return a deterministic generated collection of traffic points."""
    rng = np.random.default_rng(seed=1)
    return [(r, s, rng.uniform(0, 24*60), np.exp(rng.gamma(3, .3)))
            for _ in range(20) for s in range(10) for r in range(2)]


def test_measure_traffic_daily(traffic_instances):
    """function should return mean traffic conditions for traveling
    buses in a trial."""
    traffic = fb.experiments.TrafficModel(None)
    for t in traffic_instances:
        traffic.fix(*t)
    result = fb.measure.measure_traffic_daily(traffic)
    expected = np.mean([val for route, stop, time, val
                        in traffic_instances])
    assert result == approx(expected)


def test_measure_traffic_hourly(traffic_instances):
    """function should return mean traffic conditions, as experienced
    within each hour."""
    traffic = fb.experiments.TrafficModel(None)
    for t in traffic_instances:
        traffic.fix(*t)
    results = [fb.measure.measure_traffic_range(traffic,
                                                60 * i,
                                                60 * (i + 1))
               for i in range(24)]
    expected = [np.mean([val for route, stop, time, val
                         in traffic_instances
                         if 60 * i <= time < 60 * (i + 1)])
                for i in range(24)]
    assert results == approx(expected)


@pytest.fixture(params=[
    ([(0, 2, 1, 1, 1)],
     [[3], [10]],
     [(3 + 7 + 10 + 16) / 8, 31 / 8, 11 / 8, 0, 8]),
])
def experiment(request):
    transfers, schedule, expected = request.param
    experiment_ = fb.experiments.Experiment(
        fb.experiments.Routes(
            routes=[3, 3],
            distance=[[1] * 3, [1] * 3],
            traffic=fb.randomvar.Fixed(1),
            demand_loading=fb.randomvar.FixedAlternating(
                [[[2, 0], [2, 0], [0, 0]],
                 [[2, 0], [2, 0], [0, 0]]]),
            demand_unloading=fb.randomvar.Fixed(
                [[0, 1, 3],
                 [0, 2, 3]]),
            transfers=transfers),
        time_loading=fb.randomvar.Fixed(1),
        time_unloading=fb.randomvar.Fixed(1),
        schedule=schedule,
        speed=1,
        headers=fb.experiments.Headers.SIMPLE)
    experiment_.expected = expected
    return experiment_


@pytest.mark.xfail(reason='need to get unloading demand working first',
                   strict=True)
def test_measure_deterministic_transfers(monkeypatch, experiment, StaticBinomialRng):
    """Tests that an experiment with transfers is measured correctly."""
    trial = fb.trial.Trial(experiment)
    monkeypatch.setattr(trial, 'rng', StaticBinomialRng())
    events = trial.simulate()
    print('\n'.join(str(event) for event in events if event.route == 1))
    results = fb.measure.measure(events, experiment.headers)
    assert list(results) == experiment.expected


def test_measure_passengers_with_transfers():
    """Transfer passengers should not get double counted."""
    events = [
        Event(10, 2, 'load', 0, 0, 10, 2, 0),
        Event(12, 1, 'depart', 0, 0, 10, 2, 0),
        Event(13, 1, 'unload', 0, 1, 10, 1, 0),
        Event(14, 1, 'transfer', 0, 1, 10, 0, 1),
        Event(16, 2, 'load', 1, 0, 16, 2, 0),
    ]
    results = fb.measure.measure(events, fb.experiments.Headers.SIMPLE)
    assert results[-1] == 3
