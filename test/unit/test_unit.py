"""Collected unit tests."""

import csv

import numpy as np

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
    assert (batch == np.array([[1., 3., 0., 1.], [1., 3., 0., 1.], [1., 3., 0., 1.]])).all()

def test_measure(deterministic_experiment):
    """Tests that measure takes a list of events and returns a single
    set of random variables."""
    expected_loading = 4
    expected_moving = 8
    expected_holding = 0
    expected_passengers = 2
    expected = np.array([
        expected_loading, expected_moving,
        expected_holding, expected_passengers])
    events = [
            Event(10, 0, 'unload', 0, 0, 10, 0),
            Event(10, 2, 'load', 0, 0, 10, 2),
            Event(12, 0, 'load', 0, 0, 10, 0),
            Event(12, 4, 'depart', 0, 0, 10, 0),
            Event(16, 2, 'unload', 0, 1, 10, -2),
            Event(18, 0, 'load', 0, 1, 10, 0),
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
    intervals = fb.main.confidence_interval(trials, np.random.default_rng())
    assert intervals.shape == (3,2)
    assert (intervals[0] < intervals[1]).all()
    assert (intervals[1] < intervals[2]).all()
    assert (intervals[:,0] <= intervals[:,1]).all()
