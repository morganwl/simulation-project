"""Test that experiments are generating consistent results."""

import numpy as np
import pytest

from freebus.types import Event
from freebus.experiments import Experiment
from freebus.randomvar import Fixed, Pois
from freebus.trial import Trial, Bus
import freebus as fb

def two_stop():
    """Returns a simple two-stop system."""
    return Experiment(
            routes = [2],
            distance = [[1, 0]],
            traffic = Fixed(1),
            demand_loading = Pois([[1, 0]]),
            demand_unloading = Pois([[0, 1]]),
            time_loading = Fixed(.01),
            time_unloading = Fixed(.1),
            schedule = [[1]],
            headers=['loading-time', 'moving-time', 'holding-time', 'total-passengers']
            )

def test_two_stop():
    """The two-stop experiment should have an average of 1 passenger a
    day. The loading time does not lend itself to simple analysis."""
    experiment = two_stop()
    results = fb.main.simulate_batch(experiment, 10000)
    means = np.mean(results, axis=0)
    assert means[3] == pytest.approx(1, rel=0.1)
    assert means[1] == pytest.approx(3, rel=0.1)

def test_fixed_loading():
    """A fixed loading time should yield a consistent duration for loading events."""
    experiment = two_stop()
    trial = Trial(experiment)
    bus = Bus(0, 0, 1)
    event = trial.generate_event_load(bus)
    print(event)
    assert event.dur == 0.01 * event.passengers
    trials = np.fromiter((Trial(experiment).generate_event_load(Bus(0,0,1)).passengers
            for _ in range(10000)), dtype=np.float64)
    assert np.mean(trials) == pytest.approx(1, rel=0.1)
    trials = np.fromiter((Trial(experiment).generate_event_load(Bus(0,0,1)).dur
            for _ in range(10000)), dtype=np.float64)
    assert np.mean(trials) == pytest.approx(.01, rel=0.05)

def test_empty_loading():
    """A stop with zero loading demand should load zero passengers."""
    experiment = two_stop()
    bus = Bus(0, 1, 4)
    bus.passengers = 1
    trials = np.fromiter((Trial(experiment).generate_event_load(bus).passengers
            for _ in range(10000)), dtype=np.float64)
    assert np.mean(trials) == pytest.approx(0, rel=0.1)
    bus = Bus(0, 1, 4)
    bus.passengers = 1
    trials = np.fromiter((Trial(experiment).generate_event_load(bus).dur
            for _ in range(10000)), dtype=np.float64)
    assert np.mean(trials) == pytest.approx(0, rel=0.01)

def test_last_stop_unloading():
    """All passengers should unload at the last stop."""
    experiment = two_stop()
    def unload_last_stop(experiment):
        bus = Bus(0, 0, 1)
        bus.stop, bus.time, bus.passengers = 1, 4, 1
        return Trial(experiment).generate_event_unload(bus).passengers
    assert np.mean(
            np.fromiter(
                (unload_last_stop(experiment) for _ in range(1000)),
                dtype=np.float64)) == -1

def test_measure_unload():
    experiment = two_stop()
    events = [
            Event(1, .01, 'load', 0, 0, 1, 1),
            Event(4.01, 1, 'unload', 0, 1, 1, -1)]
    measurement = fb.main.measure(events, experiment.headers)
    assert measurement.tolist() == [0.01, 0, 0, 1]
