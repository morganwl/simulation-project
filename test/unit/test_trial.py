"""Test the process of simulating a single trial."""

import numpy as np

import freebus as fb
from freebus.types import Event
from freebus.trial import Trial, simulate, Bus

def test_simulate(deterministic_experiment):
    """Tests that a simple simulation returns a list of events."""
    events = simulate(deterministic_experiment)
    expected = [
            Event(10, 0, 'unload', 0, 0, 10, 0),
            Event(10, 5., 'wait', 0, 0, 10, 0, 1),
            Event(10, 1, 'load', 0, 0, 10, 1),
            Event(11, 0, 'wait', 0, 0, 10, 1),
            Event(11, 3., 'depart', 0, 0, 10, 1),
            Event(14., 1, 'unload', 0, 1, 10, 0),
            Event(15., 0., 'wait', 0, 1, 10, 0, 0),
            Event(15., 0., 'depart', 0, 1, 10, 0)]
    assert events == expected


def test_generate_event_unload(deterministic_experiment):
    """Tests that an unload event with simple parameters returns an
    event with appropriate results."""
    bus = Bus(0, 0, 10)
    bus.time = 15
    bus.stop = 1
    bus.passengers = 5
    trial = Trial(deterministic_experiment)
    event = trial.generate_event(bus)
    expected = Event(15, 5, 'unload', 0, 1, 10, 0)
    assert event == expected
    assert bus.state == 'wait'


def test_generate_event_wait(deterministic_experiment):
    """Tests that a wait event with simple parameters returns an event
    with appropriate results."""
    expected = Event(10, 5, 'wait', 0, 0, 10, 0, 1)
    bus = Bus(0, 0, 10)
    bus.time = 10
    bus.state = 'wait'
    trial = Trial(deterministic_experiment)
    event = trial.generate_event(bus)
    assert event == expected
    assert bus.state == 'load'


def test_generate_event_load(deterministic_experiment):
    """Tests that an load event with simple parameters returns an
    event with appropriate results."""
    expected = Event(10, 1, 'load', 0, 0, 10, 1)
    bus = Bus(0, 0, 10)
    bus.state = 'load'
    trial = Trial(deterministic_experiment)
    trial.stops[bus.route][bus.stop].waiting = 1
    event = trial.generate_event(bus)
    assert event == expected
    assert bus.state == 'wait'


def test_generate_event_depart_last(deterministic_experiment):
    """Tests that a depart event will set a bus to inactive at the last
    stop."""
    expected = Event(16, 0, 'depart', 0, 1, 10, 0)
    bus = Bus(0, 0, 10)
    bus.state = 'depart'
    bus.stop = 1
    bus.time = 16
    trial = Trial(deterministic_experiment)
    event = trial.generate_event(bus)
    assert event == expected
    assert bus.active is False
