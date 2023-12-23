"""Test the process of simulating a single trial."""

import numpy as np

import pytest
from pytest import approx

import freebus as fb
from freebus.types import Event
from freebus.trial import Trial, simulate, Bus


def two_routes():
    return fb.experiments.Experiment(
        routes=fb.experiments.Routes(
            [2, 2],
            distance=[[1, 0], [2, 0]],
            traffic=fb.randomvar.Fixed(1),
            demand_loading=fb.randomvar.FixedAlternating(
                [[[1, 0], [0, 0]], [[1, 0], [0, 0]]]),
            demand_unloading=fb.randomvar.Fixed([[0, 2], [0, 2]]),),
        time_loading=fb.randomvar.Fixed(1),
        time_unloading=fb.randomvar.Fixed(1),
        schedule=[[10, 15], [10, 20]],
        headers=['waiting-time', 'loading-time', 'moving-time',
                 'holding-time', 'total-passengers']
    )


@pytest.mark.parametrize('rand_func', [fb.randomvar.Gamma(2, .5),
                                       fb.randomvar.Gamma(4, .25),
                                       fb.randomvar.Gamma(1, 1)])
@pytest.mark.parametrize('time_func', [fb.randomvar.Fixed(.5),
                                       fb.randomvar.Fixed(.25),
                                       fb.randomvar.BetaTimeFunc(2, 2, pdf=True)])
def test_traffic_reset(rand_func, time_func):
    """Tests that the traffic model is reset at the start of every trial."""
    experiment = fb.experiments.Experiment(
        routes=fb.experiments.Routes(
            [2],
            distance=[[1, 0]],
            traffic=fb.experiments.TrafficModel(rand_func, time_func),
            demand_loading=fb.randomvar.Fixed([0]),
            demand_unloading=fb.randomvar.Fixed(1)),
        time_loading=fb.randomvar.Fixed(1),
        time_unloading=fb.randomvar.Fixed(1),
        schedule=[[360]],
        headers=['waiting-time', 'loading-time', 'moving-time',
                 'holding-time', 'total-passengers']
    )
    trials = [Trial(experiment).simulate() for _ in range(3)]
    departures = [[e for e in t if e.etype == 'depart'] for t in trials]
    first_departure_times = np.array([d[0].dur for d in departures])
    assert first_departure_times != approx([first_departure_times[0]
                                            for _ in range(3)])


def test_traffic_no_reset():
    """Tests that simultaneous travel times are the same within the same
    travel."""
    experiment = fb.experiments.Experiment(
        routes=fb.experiments.Routes(
            [2],
            distance=[[1, 0]],
            traffic=fb.experiments.TrafficModel(fb.randomvar.Fixed(.5)),
            demand_loading=fb.randomvar.Fixed(0),
            demand_unloading=fb.randomvar.Fixed(1)),
        time_loading=fb.randomvar.Fixed(1),
        time_unloading=fb.randomvar.Fixed(1),
        schedule=[[360, 360, 360]],
        headers=['waiting-time', 'loading-time', 'moving-time',
                 'holding-time', 'total-passengers']
    )
    events = Trial(experiment).simulate()
    first_departure_times = [e.dur for e in events
                             if e.etype == 'depart'
                             and e.stop == 0]
    assert first_departure_times == approx([first_departure_times[0]
                                            for _ in range(3)])


def test_two_routes():
    """Tests that an experiment with multiple routes generates the
    expected set of events."""
    trial = Trial(two_routes())
    events = trial.simulate()
    expected = {
            Event(10, 0, 'unload', 0, 0, 10, 0),
            Event(10, 5., 'wait', 0, 0, 10, 0, 1),
            Event(10, 1, 'load', 0, 0, 10, 1),
            Event(11, 0, 'wait', 0, 0, 10, 1),
            Event(11, 3., 'depart', 0, 0, 10, 1),
            Event(14., 1, 'unload', 0, 1, 10, 0),
            Event(15., 0., 'wait', 0, 1, 10, 0, 0),
            Event(15., 0., 'depart', 0, 1, 10, 0),
            Event(10, 0, 'unload', 1, 0, 10, 0),
            Event(10, 5., 'wait', 1, 0, 10, 0, 1),
            Event(10, 1, 'load', 1, 0, 10, 1),
            Event(11, 0, 'wait', 1, 0, 10, 1),
            Event(11, 6., 'depart', 1, 0, 10, 1),
            Event(17., 1, 'unload', 1, 1, 10, 0),
            Event(18., 0., 'wait', 1, 1, 10, 0, 0),
            Event(18., 0., 'depart', 1, 1, 10, 0),
            Event(15, 0, 'unload', 0, 0, 15, 0),
            Event(15, 2, 'wait', 0, 0, 15, 0, 1),
            Event(15, 1, 'load', 0, 0, 15, 1),
            Event(16, 0, 'wait', 0, 0, 15, 1),
            Event(16, 3., 'depart', 0, 0, 15, 1),
            Event(19., 1, 'unload', 0, 1, 15, 0),
            Event(20., 0., 'wait', 0, 1, 15, 0, 0),
            Event(20., 0., 'depart', 0, 1, 15, 0),
            Event(20, 0, 'unload', 1, 0, 20, 0),
            Event(20, 4.5, 'wait', 1, 0, 20, 0, 1),
            Event(20, 1, 'load', 1, 0, 20, 1),
            Event(21, 0, 'wait', 1, 0, 20, 1),
            Event(21, 6., 'depart', 1, 0, 20, 1),
            Event(27., 1, 'unload', 1, 1, 20, 0),
            Event(28., 0., 'wait', 1, 1, 20, 0, 0),
            Event(28., 0., 'depart', 1, 1, 20, 0),
    }
    assert set(events) == expected


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


def two_routes_with_transfer():
    """Returns a simple two-route experiment with one transfer."""
    return fb.experiments.Experiment(
        routes=fb.experiments.Routes(
            [2, 2],
            distance=[[1, 0], [2, 0]],
            traffic=fb.randomvar.Fixed(1),
            demand_loading=fb.randomvar.FixedAlternating(
                [[[1, 0], [0, 0]], [[1, 0], [0, 0]]]),
            demand_unloading=fb.randomvar.Fixed([[0, 2], [0, 2]]),
            transfers=[(0, 1, 1, 0, 1),]),
        time_loading=fb.randomvar.Fixed(1),
        time_unloading=fb.randomvar.Fixed(1),
        schedule=[[10, 15], [10, 20]],
        headers=['waiting-time', 'loading-time', 'moving-time',
                 'holding-time', 'total-passengers']
    )


def test_generate_event_unload_before_transfer(monkeypatch, StaticBinomialRng):
    """An unload event should not contain passengers transferring to
    another route."""
    experiment = two_routes_with_transfer()
    trial = Trial(experiment)

    class FixedOutcome:
        """uniform method iterates over a fixed list of outcomes."""
        def __init__(self, outcomes: list[float]):
            self.outcomes = iter(outcomes)

        def uniform(self):
            """Returns the next predetermined uniform random
            variable."""
            return next(self.outcomes)
    monkeypatch.setattr(trial, 'rng', StaticBinomialRng())
    bus = Bus(0, 0, 10)
    bus.state = 'unload'
    bus.stop = 1
    bus.time = 14
    bus.passengers = 2
    event = trial.generate_event(bus)
    expected = Event(
        time=14,
        dur=1,
        etype='unload',
        route=0,
        stop=1,
        busid=10,
        passengers=1,
        waiting=0)
    assert event == expected


def test_generate_event_transfer(monkeypatch, StaticBinomialRng):
    """A transfer between two stops should generate a transfer event for
    the target stop."""
    experiment = two_routes_with_transfer()
    trial = Trial(experiment)
    monkeypatch.setattr(trial, 'rng', StaticBinomialRng())
    bus = Bus(0, 0, 10)
    bus.state = 'transfer'
    bus.stop = 1
    bus.time = 15
    bus.passengers = 1
    event = trial.generate_event(bus)
    expected = Event(
        time=15,
        dur=1,
        etype='transfer',
        route=0,
        stop=1,
        busid=10,
        passengers=0,
        waiting=1)
    assert event == expected


def test_generate_second_event_transfer(monkeypatch, StaticBinomialRng):
    """The duration of a transfer event should not include time spent by
    passengers waiting from an earlier transfer."""
    experiment = two_routes_with_transfer()
    trial = Trial(experiment)
    transfer, *_ = experiment.get_transfers(0, 1)
    monkeypatch.setattr(trial, 'rng', StaticBinomialRng())
    bus1 = Bus(0, 0, 10)
    bus1.state = 'transfer'
    bus1.stop = 1
    bus1.time = 15
    bus1.passengers = 1
    bus2 = Bus(0, 0, 15)
    bus2.state = 'transfer'
    bus2.stop = 1
    bus2.time = 20
    bus2.passengers = 1
    trial.generate_event(bus1)
    event = trial.generate_event(bus2)
    assert transfer.waiting == 2
    assert event.waiting == 2
    assert event.dur == 1


@pytest.fixture
def two_routes_with_transfer_experiment():
    """A system with two binary routes, with fixed passenger rates and
    loading times, and a transfer from route 0 to route 1."""
    return fb.experiments.Experiment(
        routes=fb.experiments.Routes(
            [2, 2],
            distance=[[1, 0], [2, 0]],
            traffic=fb.randomvar.Fixed(1),
            demand_loading=fb.randomvar.FixedAlternating(
                [[[1, 0], [0, 0]], [[1, 0], [0, 0]]]),
            demand_unloading=fb.randomvar.Fixed([[0, 2], [0, 2]]),
            transfers=[(0, 1, 1, 0, 1),]),
        time_loading=fb.randomvar.Fixed(1),
        time_unloading=fb.randomvar.Fixed(1),
        schedule=[[10, 15], [10, 20]],
        headers=['waiting-time', 'loading-time', 'moving-time',
                 'holding-time', 'total-passengers']
    )


@pytest.fixture
def transfer_trial(two_routes_with_transfer_experiment):
    return Trial(two_routes_with_transfer_experiment)


def test_wait_clears_transfer(transfer_trial, monkeypatch):
    """A wait event should absorb any passengers waiting from a
    transfer."""
    trial = transfer_trial
    # don't generate any new passengers
    monkeypatch.setattr(trial.experiment._routes, 'demand_loading',
                        fb.randomvar.Fixed(0))
    bus = Bus(1, 0, 10, state='wait')
    transfer, *_ = trial.experiment.get_transfers_to(1, 0)
    transfer.waiting = 1
    transfer.last_time = 8
    event = trial.generate_event(bus)
    expected = Event(10, 2, 'wait', 1, 0, 10, 0, 1)
    assert event == expected
    assert transfer.waiting == 0


@pytest.mark.parametrize(['buses', 'expected',],
                         [([Bus(0, 1, 5, passengers=2, state='unload'),
                            Bus(1, 0, 10, passengers=0, state='unload')],
                           Event(12, 0, 'wait', 1, 0, 10, 2)),
                          ([Bus(0, 1, 5, passengers=2, state='unload'),
                            Bus(0, 1, 10, passengers=2, state='unload'),
                            Bus(1, 0, 15, passengers=0, state='unload')],
                           Event(18, 0, 'wait', 1, 0, 15, 3))])
def test_load_from_transfer(monkeypatch, StaticBinomialRng, transfer_trial, buses, expected):
    """A bus should load passengers waiting after a transfer, as well as
    passengers starting at a given stop."""
    monkeypatch.setattr(transfer_trial, 'rng', StaticBinomialRng())
    for bus in buses:
        while bus.state != 'depart':
            event = transfer_trial.generate_event(bus)
            print(event)
    assert event == expected


@pytest.mark.parametrize(['waiting', 'last_time'],
                         [(1, 8),
                          (2, 8),
                          (2, 7)])
@pytest.mark.parametrize('bus_passengers', [0, 1, 2])
def test_transfer_wait(transfer_trial, waiting, last_time,
                       bus_passengers):
    """Passengers already waiting from a transfer should have their wait
    time recorded between transfer events."""
    bus = Bus(0, 1, 10, state='unload', passengers=bus_passengers)
    transfer, *_ = transfer_trial.experiment.get_transfers(bus.route, bus.stop)
    transfer.waiting, transfer.last_time = waiting, last_time
    transfer_trial.generate_event(bus)
    event = transfer_trial.generate_event(bus)
    expected = Event(bus.time, bus.time - last_time, 'transfer_wait',
                     0, 1, 10, bus.passengers, waiting)
    assert event == expected
