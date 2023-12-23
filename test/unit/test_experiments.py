"""Test the Experiment object class."""

from copy import deepcopy

import pytest
from pytest import approx
import numpy as np

from freebus.experiments import Experiment, Routes, TrafficModel, \
    get_builtin_experiments
from freebus.randomvar import Fixed, FixedAlternating, Pois, BetaTimeFunc, \
    Gamma
import freebus as fb



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
            Pois([[0, 1]]),),
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


def test_experiment_repr_gather_pert():
    """The gather Pert option does not include loading time information
    in the experiment __repr__."""
    builtins = get_builtin_experiments()
    short = builtins['b35-short']
    long = builtins['b35-long']
    short.gather_pert = True
    long.gather_pert = True
    assert repr(short) == repr(long)


def test_experiment_repr_builtins():
    """Tests that all builtin experiments have a stable
    representation."""
    builtins = get_builtin_experiments()
    copies = {k: deepcopy(e) for k, e in builtins.items()}
    for k, b in builtins.items():
        assert b != copies[k]
    assert str(copies) == str(builtins)


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


@pytest.mark.parametrize('rand_func', [Gamma(2, .5), Gamma(4, .25),
                                       Gamma(1, 1)])
@pytest.mark.parametrize('time_func', [Fixed(.5), Fixed(.25),
                                       BetaTimeFunc(2, 2, pdf=True)])
def test_traffic_model_consistent(rand_func, time_func):
    """Multiple queries of traffic with the same parameters should yield
    the same results."""
    traffic = TrafficModel(rand_func, time_func)
    route, stop, time = 0, 1, 5
    results = np.array([traffic(route, stop, time) for _ in range(10)])
    assert (results[0] == results).all()


@pytest.mark.parametrize('rand_func', [Gamma(2, .5), Gamma(4, .25),
                                       Gamma(1, 1)])
@pytest.mark.parametrize('time_func', [Fixed(.5), Fixed(.25),
                                       BetaTimeFunc(2, 2, pdf=True)])
def test_traffic_model_reset(rand_func, time_func):
    """Results for the same query should be different after a reset."""
    traffic = TrafficModel(rand_func, time_func)
    route, stop, time = 0, 1, 10
    shape = 100
    results = np.zeros(shape)
    for i in range(shape):
        results[i] = traffic(route, stop, time)
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


@pytest.mark.parametrize(['fixed', 'close', 'far'],
                         [[(0, 1, 250), (0, 1, 260), (0, 1, 275)]])
@pytest.mark.parametrize('rand_func', [fb.randomvar.Beta(2, 2)])
def test_traffic_model_smooth(fixed, close, far, rand_func):
    """Results should be affected by earlier queries to nearby
    parameters."""
    traffic = TrafficModel(rand_func)
    trials = 100
    fixed_val = 1.75

    def measure_and_reset(traffic, point, fixed_point=None, fixed_val=None):
        if fixed_point is not None:
            traffic.fix(*fixed_point, fixed_val)
        result = traffic(*point)
        traffic.reset()
        return result
    random_results = np.array([measure_and_reset(traffic, close)
                               for _ in range(trials)])
    close_results = np.array([measure_and_reset(traffic, close,
                                                fixed, fixed_val)
                              for _ in range(trials)])
    far_results = np.array([measure_and_reset(traffic, far,
                                              fixed, fixed_val)
                            for _ in range(trials)])
    random_variance = np.mean((random_results - fixed_val)**2)
    close_variance = np.mean((close_results - fixed_val)**2)
    far_variance = np.mean((far_results - fixed_val)**2)
    assert close_variance < random_variance
    assert close_variance < far_variance
    assert far_variance < random_variance


@pytest.mark.parametrize('time', [60 * 4 * i for i in range(24 // 4)])
@pytest.mark.parametrize('mean', [.5 * (i+1) for i in range(4)])
def test_traffic_model_constant(time, mean):
    """A constant traffic model should report the same distribution for
    any time of day. Because traffic has a lower bound of 1, the
    magnitude of traffic can be understood as traffic(...) - 1."""
    traffic = TrafficModel(lambda: 1, time_func=Fixed(mean))
    result = traffic(0, 0, time)
    assert result == 1 + mean


@pytest.mark.parametrize('time', [60 * 4 * i for i in range(24 // 4)])
@pytest.mark.parametrize('funcs', [[BetaTimeFunc(3, 4, pdf=True)]])
def test_traffic_model_beta(time, funcs):
    """A beta-shaped traffic model should report a distribution centered
    on the sum of one or more beta distributions for any given time."""

    def fresh_traffic(*args):
        """Instantiate a fresh TrafficModel and generate a traffic
        value."""
        traffic = TrafficModel(Gamma(10, .1),
                               time_func=lambda t: sum(f(t) for f in funcs))
        return traffic(*args)
    result = np.mean([fresh_traffic(0, 0, time) for _ in range(250)])
    assert result == approx(1 + sum(f(time) for f in funcs), rel=0.1)


@pytest.mark.parametrize('time', [60 * 4 * i for i in range(24 // 4)])
@pytest.mark.parametrize('time_func', [BetaTimeFunc(3, 6, pdf=True)],)
def test_traffic_daily_scale(time, time_func, ReturnFrom):
    """A daily scale rv should be generated once per day and scale all
    results for that day exponentially."""
    daily_scales = [.75, 1, 1.25]
    daily_scale_func = ReturnFrom(daily_scales)
    traffic = TrafficModel(Gamma(20, .05, seed=1),
                           time_func=time_func,
                           daily_func=daily_scale_func)
    results = []
    for _ in daily_scales:
        results.append(np.mean([traffic(i, 0, time) for i in range(2000)]))
        traffic.reset()
    assert results == approx([(1 + time_func(time)) ** ds
                              for ds in daily_scales], rel=0.1)
