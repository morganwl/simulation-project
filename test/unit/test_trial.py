"""Test the process of simulating a single trial."""

import numpy as np

import freebus as fb
from freebus.types import Event
from freebus.trial import Trial, simulate, Bus

def test_simulate(deterministic_experiment):
    """Tests that a simple simulation returns a list of events."""
    expected = [
            (10, 0, 'unload', 0, 0, 10, 0),
            (10, 1, 'load', 0, 0, 10, 1),
            (11, 0, 'load', 0, 0, 10, 0),
            (11, 3, 'depart', 0, 0, 10, 0),
            (14, 1, 'unload', 0, 1, 10, -1),
            (15, 0, 'load', 0, 1, 10, 0),
            (15, 0, 'depart', 0, 1, 10, 0)]
    events = simulate(deterministic_experiment)
    assert events == expected

def test_generate_event_unload(deterministic_experiment):
    """Tests that an unload event with simple parameters returns an
    event with appropriate results."""
    expected = Event(15, 5, 'unload', 0, 1, 10, -5)
    bus = Bus(0, 0, 10)
    bus.time = 15
    bus.stop = 1
    bus.passengers = 5
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
    event = trial.generate_event(bus)
    assert event == expected
    assert bus.state == 'load'

def test_generate_event_depart_last(deterministic_experiment):
    """Tests that an apart event will set a bus to inactive at the last stop."""
    expected = (16, 0, 'depart', 0, 1, 10, 0)
    bus = Bus(0, 0, 10)
    bus.state = 'depart'
    bus.stop = 1
    bus.time = 16
    trial = Trial(deterministic_experiment)
    event = trial.generate_event(bus)
    assert event == expected
    assert bus.active is False
