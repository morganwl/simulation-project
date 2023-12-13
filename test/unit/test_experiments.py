"""Test the Experiment object class."""

from freebus.experiments import Experiment, Routes, TrafficModel
from freebus.randomvar import Fixed, FixedAlternating, Pois

import numpy as np


def test_experiment_repr_simple():
    """Tests that an Experiment object has a stable, useful,
    representation."""
    args1 = [
        Routes(
            [2], [1, 0],
            Fixed(1),
            FixedAlternating([[[1, 0], [0, 0]]]),
            Fixed([[0, 1]]),),
        Fixed(1),
        Fixed(1),
        [[0, 10]],
        ['loading-time', 'moving-time', 'holding-time', 'total-passengers'],
    ]
    args2 = args1[:]
    args2[-2] = [[0, 10, 20]]
    exp1 = Experiment(*args1)
    exp2 = Experiment(*args1)
    exp3 = Experiment(*args2)
    assert exp1 != exp2
    assert str(exp1) == str(exp2)
    assert str(exp1) != str(exp3)


def test_experiment_repr_poisson():
    """Tests that an Experiment object has a stable representation when
    containing Poisson parameters."""
    args1 = [
        Routes(
            [2], [1, 0],
            Fixed(1),
            Pois([[1, 0]]),
            Pois([[Fixed(0), Fixed(1)]]),),
        Fixed(1),
        Fixed(1),
        [[0, 10]],
        ['loading-time', 'moving-time', 'holding-time', 'total-passengers'],
    ]
    args2 = args1[:]
    args2[-2] = [[0, 10, 20]]
    exp1 = Experiment(*args1)
    exp2 = Experiment(*args1)
    exp3 = Experiment(*args2)
    assert exp1 != exp2
    assert str(exp1) == str(exp2)
    assert str(exp1) != str(exp3)


def test_experiment_checksum():
    """Tests that an experiment yields a stable checksum."""
    args1 = [
        Routes(
            [2], [1, 0],
            Fixed(1),
            FixedAlternating([[[1, 0], [0, 0]]]),
            Fixed([[0, 1]]),),
        Fixed(1),
        Fixed(1),
        [[0, 10]],
        ['loading-time', 'moving-time', 'holding-time', 'total-passengers'],
    ]
    args2 = args1[:]
    args2[-2] = [[0, 10, 20]]
    exp1 = Experiment(*args1)
    exp2 = Experiment(*args1)
    exp3 = Experiment(*args2)
    assert exp1 != exp2
    assert exp1.checksum() == exp2.checksum()
    assert exp1.checksum() != exp3.checksum()


def test_traffic_model_consistent():
    """Multiple queries of traffic with the same parameters should yield
    the same results."""
    traffic = TrafficModel(Fixed(.5))
    route, stop, time = 0, 1, 5
    results = np.array([traffic(route, stop, time) for _ in range(10)])
    assert (results[0] == results).all()


def test_traffic_model_reset():
    """Results for the same query should be different after a reset."""
    traffic = TrafficModel(Fixed(.5))
    route, stop, time = 0, 1, 5
    shape = 5
    results = np.zeros(shape)
    for i in range(shape):
        results[i] = (traffic(route, stop, time))
        traffic.reset()
    assert (results[0] != results).any()


def test_find_neighors():
    """The traffic model should return the correct neighbors."""
    traffic = TrafficModel(Fixed(.5))
    traffic.fix(0, 0, 1, .25)
    traffic.fix(0, 0, 2, .75)
    traffic.fix(0, 0, 3, .8)
    earlier, later = traffic.find_neighbors(0, 0, 1.5)
    assert earlier.val == .25
    assert later.val == .75


def test_traffic_model_smooth():
    """Results should be affected by earlier queries to nearby
    parameters."""
    traffic = TrafficModel(Fixed(.5))
    r0, s0, t0 = 0, 1, 5
    r1, s1, t1 = 0, 1, 6
    r2, s2, t2 = 0, 1, 10
    shape = 100
    random_results = np.zeros(shape)
    close_results = np.zeros(shape)
    far_results = np.zeros(shape)
    for i in range(shape):
        random_results[i] = traffic(r1, s1, t1)
        traffic.reset()
        traffic.fix(r0, s0, t0, .75)
        close_results[i] = traffic(r1, s1, t1)
        traffic.reset()
        traffic.fix(r0, s0, t0, .75)
        far_results[i] = traffic(r2, s2, t2)
        traffic.reset()
    random_variance = np.mean((random_results - .75)**2)
    close_variance = np.mean((close_results - .75)**2)
    far_variance = np.mean((far_results - .75)**2)
    assert close_variance < random_variance
    assert close_variance < far_variance
    assert far_variance < random_variance
