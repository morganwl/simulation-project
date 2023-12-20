"""Tests generation of measurements from a single trial."""

import numpy as np
import pytest
from pytest import fixture, approx

import freebus as fb


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
    assert result == expected


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
