"""Collected unit tests."""

import csv

import numpy as np

import freebus as fb

def deterministic_experiment_stub():
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
    print(expected)
    with open(output, newline='', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile)
        result = [next(reader)]
        result.extend([[float(v) for v in row] for row in reader])
        assert expected == result

def test_simulate_batch(monkeypatch):
    """Tests returning a batch of simulation results."""
    class Experiment:
        # pylint: disable=too-few-public-methods
        """Experiment stub."""
        headers = ['a', 'b']
    monkeypatch.setattr(fb.main, 'simulate', lambda x: np.ones(len(x.headers)))
    monkeypatch.setattr(fb.main, 'measure', lambda x, y: x)
    batch = fb.main.simulate_batch(Experiment, 3)
    assert (batch == np.array([[1., 1.], [1., 1.], [1., 1.]])).all()

def test_simulate():
    """Tests that a simple simulation returns a list of events."""
    experiment = deterministic_experiment_stub()
    expected = [
            (10, 0, 'unload', 0, 0, 10, 0),
            (10, 1, 'load', 0, 0, 10, 1),
            (11, 0, 'load', 0, 0, 10, 0),
            (11, 3, 'depart', 0, 0, 10, 0),
            (14, 1, 'unload', 0, 1, 10, -1),
            (15, 0, 'load', 0, 1, 10, 0),
            (15, 0, 'depart', 0, 1, 10, 0)]
    events = fb.main.simulate(experiment)
    assert events == expected

def test_generate_event_unload():
    """Tests that an unload event with simple parameters returns an
    event with appropriate results."""
    expected = (15, 5, 'unload', 0, 1, 10, -5)
    experiment = deterministic_experiment_stub()
    stops = [[None, fb.main.Stop()]]
    bus = fb.main.Bus(0, 0, 10)
    bus.time = 15
    bus.stop = 1
    bus.passengers = 5
    event = fb.main.generate_event(bus, stops, experiment)
    assert event == expected
    assert bus.state == 'load'

def test_generate_event_load():
    """Tests that an load event with simple parameters returns an
    event with appropriate results."""
    expected = (10, 1, 'load', 0, 0, 10, 1)
    experiment = deterministic_experiment_stub()
    stops = [[fb.main.Stop(), None]]
    bus = fb.main.Bus(0, 0, 10)
    bus.state = 'load'
    event = fb.main.generate_event(bus, stops, experiment)
    assert event == expected
    assert bus.state == 'load'

def test_generate_event_depart_last():
    """Tests that an apart event will set a bus to inactive at the last stop."""
    expected = (16, 0, 'depart', 0, 1, 10, 0)
    experiment = deterministic_experiment_stub()
    stops = [[None, fb.main.Stop()]]
    bus = fb.main.Bus(0, 0, 10)
    bus.state = 'depart'
    bus.stop = 1
    bus.time = 16
    event = fb.main.generate_event(bus, stops, experiment)
    assert event == expected
    assert bus.active is False
