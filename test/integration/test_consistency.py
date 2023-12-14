"""Test that experiments are generating consistent results."""

import numpy as np
import pytest

from freebus.types import Event
from freebus.experiments import Experiment, Routes
from freebus.randomvar import Fixed, Pois
from freebus.trial import Trial, Bus
import freebus as fb


def two_stop():
    """Returns a simple two-stop system."""
    return Experiment(
        Routes(
            routes=[2],
            distance=[[1, 0]],
            traffic=Fixed(1),
            demand_loading=Pois([[1, 0]]),
            demand_unloading=Pois([[0, 1]]),),
        time_loading=Fixed(.01),
        time_unloading=Fixed(.1),
        schedule=[[1]],
        headers=['waiting-time', 'loading-time', 'moving-time',
                 'holding-time', 'total-passengers']
    )


def test_two_stop():
    """The two-stop experiment should have an average of 1 passenger a
    day. The loading time does not lend itself to simple analysis."""
    experiment = two_stop()
    results = fb.main.simulate_batch(experiment, 1000)
    means = np.mean(results, axis=0)
    means = dict(zip(experiment.headers, means))
    assert means['total-passengers'] == pytest.approx(1, rel=0.1)


def wait_and_load(experiment, route, stop, t):
    """Return a load event after generating a wait event."""
    trial = Trial(experiment)
    bus = Bus(route, stop, t)
    trial.generate_event_wait(bus)
    return trial.generate_event_load(bus)


def test_fixed_loading():
    """A fixed loading time should yield a consistent duration for loading events."""
    experiment = two_stop()
    trial = Trial(experiment)
    bus = Bus(0, 0, 1)
    trial.stops[0][0].waiting = 1
    event = trial.generate_event_load(bus)
    print(event)
    assert event.dur == 0.01 * event.passengers
    trials = np.fromiter((wait_and_load(experiment, 0, 0, 1).passengers
                          for _ in range(1000)), dtype=np.float64)
    assert np.mean(trials) == pytest.approx(1, rel=0.1)
    trials = np.fromiter((wait_and_load(experiment, 0, 0, 1).dur
                          for _ in range(1000)), dtype=np.float64)
    assert np.mean(trials) == pytest.approx(.01, rel=0.1)

def test_empty_loading():
    """A stop with zero loading demand should load zero passengers."""
    experiment = two_stop()
    bus = Bus(0, 1, 4)
    bus.passengers = 1
    trials = np.fromiter((Trial(experiment).generate_event_load(bus).passengers
                          for _ in range(1000)), dtype=np.float64)
    assert np.mean(trials) == pytest.approx(1, rel=0.1)
    bus = Bus(0, 1, 4)
    bus.passengers = 1
    trials = np.fromiter((Trial(experiment).generate_event_load(bus).dur
                          for _ in range(1000)), dtype=np.float64)
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
            dtype=np.float64)) == 0


def test_measure_unload():
    """Tests that a list of load and unload events are measured correctly."""
    experiment = two_stop()
    events = [
        Event(1, .01, 'load', 0, 0, 1, 1),
        Event(4.01, 1, 'unload', 0, 1, 1, 0)]
    measurement = dict(
        zip(experiment.headers, fb.main.measure(events, experiment.headers)))
    assert measurement['loading-time'] == 0.01
    assert measurement['total-passengers'] == 1


def test_leapfrog_fixed(deterministic_experiment):
    """Tests that a bus which arrives while another bus is loading
    passengers will load the next available passengers."""
    deterministic_experiment.schedule = [[10, 10.5]]
    results = dict(
        zip(deterministic_experiment.headers,
            fb.main.simulate_batch(deterministic_experiment, 1)[0]))
    assert results['waiting-time'] == 2.625
    assert results['loading-time'] == 1.5
    assert results['total-passengers'] == 2


def test_leapfrog_fixed_events(deterministic_experiment):
    """Tests that a bus which arrives while another bus is loading
    passengers will load the next available passengers."""
    deterministic_experiment.schedule = [[10, 10.5]]
    trial = Trial(deterministic_experiment)
    events = trial.simulate()
    print(events)
    assert events == [
        Event(10, 0, 'unload', 0, 0, 10, 0),
        Event(10, 5, 'wait', 0, 0, 10, 0, 1),
        Event(10, 1, 'load', 0, 0, 10, 1),
        Event(10.5, 0, 'unload', 0, 0, 10.5, 0),
        Event(10.5, 0, 'wait', 0, 0, 10.5, 0),
        Event(10.5, 3, 'depart', 0, 0, 10.5, 0),
        Event(11, .25, 'wait', 0, 0, 10, 1, 1),
        Event(11, 1, 'load', 0, 0, 10, 2),
        Event(12, 0, 'wait', 0, 0, 10, 2),
        Event(12, 3, 'depart', 0, 0, 10, 2),
        Event(13.5, 0, 'unload', 0, 1, 10.5, 0),
        Event(13.5, 0, 'wait', 0, 1, 10.5, 0),
        Event(13.5, 0, 'depart', 0, 1, 10.5, 0),
        Event(15, 2, 'unload', 0, 1, 10, 0),
        Event(17, 0, 'wait', 0, 1, 10, 0),
        Event(17, 0, 'depart', 0, 1, 10, 0),
    ]
