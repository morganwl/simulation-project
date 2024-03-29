"""Collected unit tests."""

import csv

import numpy as np
import pytest
from pytest import approx

import freebus as fb
from freebus.types import Event


def test_write_batch(tmpdir):
    """Tests appending a batch of trials to a csv file."""
    output = tmpdir / 'test_output.csv'
    headers = ['first', 'second', 'third']
    expected = [headers[:]]
    batch = np.array([[1., 2., 3.], [4., 5., 6.]])
    expected.extend(batch.tolist())
    fb.main.write_batch(batch, headers, output)
    batch = np.array([[7., 8., 9.], [10., 11., 12.]])
    expected.extend(batch.tolist())
    fb.main.write_batch(batch, headers, output)
    with open(output, newline='', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile)
        result = [next(reader)]
        result.extend([[float(v) for v in row] for row in reader])
        assert expected == result


def test_simulate_batch(deterministic_experiment):
    """Tests returning a batch of simulation results."""
    batch = fb.main.simulate_batch(deterministic_experiment, 3)
    assert (batch == np.array(
        [[5., 1., 3., 0., 1.],
         [5., 1., 3., 0., 1.],
         [5., 1., 3., 0., 1.]]
    )).all()


def test_simulate_deterministic_experiment(deterministic_experiment):
    """The deterministic experiment should always yield the same
    sequence of events."""
    events = fb.main.simulate(deterministic_experiment)
    assert events == [
        Event(10, 0, 'unload', 0, 0, 10, 0),
        Event(10, 5, 'wait', 0, 0, 10, 0, 1),
        Event(10, 1, 'load', 0, 0, 10, 1),
        Event(11, 0, 'wait', 0, 0, 10, 1),
        Event(11, 3, 'depart', 0, 0, 10, 1),
        Event(14, 1, 'unload', 0, 1, 10, 0),
        Event(15, 0, 'wait', 0, 1, 10, 0),
        Event(15, 0, 'depart', 0, 1, 10, 0),
    ]


def test_measure_deterministic_experiment(deterministic_experiment):
    """The deterministic experiment should always yield the same
    results."""
    events = fb.main.simulate(deterministic_experiment)
    results = fb.main.measure(events, deterministic_experiment.headers)
    assert (results == np.array([5., 1., 3., 0., 1.])).all()


def test_measure_one_passenger(deterministic_experiment):
    """Tests that measure takes a list of events and returns a single
    set of random variables."""
    expected_waiting = 5
    expected_loading = 1
    expected_moving = 4
    expected_holding = 0
    expected_passengers = 1
    expected = np.array([
        expected_waiting, expected_loading, expected_moving,
        expected_holding, expected_passengers])
    events = [
        Event(10, 0, 'unload', 0, 0, 10, 0),
        Event(10, 5, 'wait', 0, 0, 10, 0, 1),
        Event(10, 1, 'load', 0, 0, 10, 1),
        Event(11, 0, 'wait', 0, 0, 10, 1),
        Event(12, 4, 'depart', 0, 0, 10, 1),
        Event(16, 1, 'unload', 0, 1, 10, 0),
        Event(17, 0, 'wait', 0, 1, 10, 0),
        Event(17, 0, 'depart', 0, 1, 10, 0)]
    result = fb.main.measure(events, deterministic_experiment.headers)
    assert (result == expected).all()


def test_measure_two_passengers(deterministic_experiment):
    """Tests that measure takes a list of events and returns a single
    set of random variables."""
    # note that all times are summed over ALL passengers
    expected_waiting = 3.75
    expected_loading = 2
    expected_moving = 4
    expected_holding = 0
    expected_passengers = 2
    expected = np.array([
        expected_waiting, expected_loading, expected_moving,
        expected_holding, expected_passengers])
    events = [
        Event(10, 0, 'unload', 0, 0, 10, 0),
        Event(10, 3.75, 'wait', 0, 0, 10, 0, 2),
        Event(10, 2, 'load', 0, 0, 10, 2),
        Event(12, 0, 'load', 0, 0, 10, 2),
        Event(12, 4, 'depart', 0, 0, 10, 2),
        Event(16, 2, 'unload', 0, 1, 10, 0),
        Event(18, 0, 'wait', 0, 1, 10, 0),
        Event(18, 0, 'depart', 0, 1, 10, 0)]
    result = fb.main.measure(events, deterministic_experiment.headers)
    assert (result == expected).all()


def test_confidence_interval():
    """Tests that confidence intervals are of the desired form."""
    trials = np.column_stack([
        np.arange(0, 20, 1),
        np.arange(100, 120, 1),
        np.arange(1000, 1020, 1),
    ])
    means = np.mean(trials, axis=0)
    assert len(means) == 3
    intervals = fb.main.confidence_interval(trials, np.random.default_rng())
    assert intervals.shape == (3, 2)
    assert intervals[0][0] < means[0] < intervals[0][1]
    assert intervals[1][0] < means[1] < intervals[1][1]
    assert intervals[2][0] < means[2] < intervals[2][1]
    assert (intervals[0] < intervals[1]).all()
    assert (intervals[1] < intervals[2]).all()
    assert (intervals[:, 0] <= intervals[:, 1]).all()


@pytest.mark.parametrize('routes', [[20, 30],
                                    [10, 50],
                                    [45, 45],
                                    [48, 42],])
@pytest.mark.parametrize('daily_rates', [[8000, 10000],
                                         [1000, 5000],
                                         [12000, 12000],
                                         [12408, 11256],])
def test_route_randomize(routes, daily_rates):
    """Confirm that randomizing a route maintains original passenger
    load."""
    rate_loading, rate_unloading = (
        fb.experiments.randomize_routes(routes, daily_rates))
    assert (np.sum(rate_loading, axis=1) == daily_rates).all()
    assert (np.sum(rate_unloading, axis=1) == daily_rates).all()
    for route, rates in zip(routes, rate_loading):
        assert (np.array(rates[route-1:]) == 0).all()
    for route, rates in zip(routes, rate_unloading):
        assert rates[0] == approx(0)
        assert (np.array(rates[route:]) == 0).all()
    for loading, unloading in zip(rate_loading, rate_unloading):
        passengers = 0
        for l, u in zip(loading, unloading):
            passengers -= u
            assert passengers >= 0
            passengers += l


def test_antithetic_batch():
    """An antithetic batch has the same mean with lower variance."""
    experiment = fb.experiments.Experiment(
        fb.experiments.Routes(
            [2], [[1, 0]],
            demand_loading=fb.randomvar.Pois(
                [[.25, 0]], daily_func=fb.randomvar.Beta(1.5, 1.5)),
            demand_unloading=fb.randomvar.Fixed([[0, 1]]),
            traffic=fb.experiments.TrafficModel(
                fb.randomvar.Fixed(2),
                daily_func=fb.randomvar.Beta(1.5, 1.5))),
        time_loading=fb.randomvar.Fixed(1),
        time_unloading=fb.randomvar.Fixed(1),
        schedule=[[10, 20, 30, 40, 50, 60]],
        headers=fb.experiments.Headers.SIMPLE)
    batch = fb.main.simulate_batch(
        experiment, 500, antithetic=True)[:, np.arange(4)]
    batch = np.sum(batch, axis=1)
    mean, std = np.mean(batch, axis=0), np.std(batch, axis=0)
    exp_batch = fb.main.simulate_batch(
        experiment, 500, antithetic=False)[:, np.arange(4)]
    exp_batch = np.sum(exp_batch, axis=1)
    exp_mean, exp_std = np.mean(exp_batch, axis=0), np.std(exp_batch, axis=0)
    assert mean == approx(exp_mean, rel=0.1)
    assert std < exp_std
