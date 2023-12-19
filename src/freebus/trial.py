"""Generate the events for an individual trial."""

from collections import deque
from dataclasses import dataclass
from heapq import heappush, heappop, heapreplace

import numpy as np

from .types import Event

SPEED = 12/60


@dataclass
class Stop:
    """State structure for a single bus stop."""
    last_load: float = 0
    waiting: int = 0


class Bus:
    """State structure for a single bus."""
    def __init__(self, route, stop, time):
        self.state = 'unload'
        self.route = route
        self.stop = stop
        self.time = time
        self.id = time
        self.passengers = 0
        self.active = True

    def __repr__(self):
        return (f'{type(self).__name__}'
                f'({self.route}, {self.stop}, {self.time}, '
                f'id={self.id}, state={self.state})')

    def __lt__(self, other):
        return self.time < other.time


class Trial:
    """Object to manage state for a single trial."""

    def __init__(self, experiment, rng=None):
        experiment.reset()
        self.experiment = experiment
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.stops = [[Stop() for _ in range(r)] for r in experiment.routes]

    def simulate(self):
        """Returns a list of simulated events."""
        events = []
        queue = []
        for i, route in enumerate(self.experiment.schedule):
            for bus in route:
                heappush(queue, Bus(i, 0, bus))
        while queue:
            bus = heappop(queue)
            event = self.generate_event(bus)
            events.append(event)
            if bus.active:
                heappush(queue, bus)
        return events

    def generate_event(self, bus):
        """Generates an event based on the current state of bus, and updates
        states."""
        if bus.state == 'unload':
            return self.generate_event_unload(bus)
        if bus.state == 'transfer':
            return self.generate_event_transfer(bus)
        if bus.state == 'wait':
            return self.generate_event_wait(bus)
        if bus.state == 'load':
            return self.generate_event_load(bus)
        if bus.state == 'depart':
            return self.generate_event_depart(bus)
        raise RuntimeError('Unknown state.')

    def generate_event_depart(self, bus):
        """Generate a depart event."""
        t = (self.experiment.distance[bus.route][bus.stop]
             / self.experiment.speed
             * self.experiment.traffic(bus.route, bus.stop, bus.time))
        event = Event(bus.time, t, 'depart', bus.route, bus.stop,
                      bus.id, bus.passengers)
        bus.time += t
        bus.stop += 1
        if bus.stop < self.experiment.routes[bus.route]:
            bus.state = 'unload'
        else:
            bus.active = False
        return event

    def generate_event_wait(self, bus):
        """Generate a passenger wait event."""
        stop = self.stops[bus.route][bus.stop]
        delta = bus.time - stop.last_load
        n = self.experiment.demand_loading(bus.route, bus.stop,
                                           bus.time, scale=delta)
        t = (self.experiment.demand_loading.sum_arrivals(n, delta)
             + delta * stop.waiting)
        stop.last_load = bus.time
        stop.waiting += n
        if n > 0:
            bus.state = 'load'
        else:
            bus.state = 'depart'
        if stop.waiting:
            t /= stop.waiting
        return Event(bus.time, t, 'wait', bus.route, bus.stop, bus.id,
                     bus.passengers, stop.waiting)

    def generate_event_load(self, bus):
        """Generate a load event."""
        n = self.stops[bus.route][bus.stop].waiting
        t = sum(self.experiment.time_loading(bus.passengers + i)
                for i in range(n))
        bus.passengers += n
        self.stops[bus.route][bus.stop].waiting -= n
        event = Event(bus.time, t, 'load', bus.route, bus.stop, bus.id,
                      bus.passengers)
        bus.time += t
        bus.state = 'wait'
        return event

    def generate_event_unload(self, bus):
        """Generates an unload event."""
        rate = self.experiment.demand_unloading.expected(bus.route,
                                                         bus.stop,
                                                         bus.time)
        total_rate = sum(self.experiment.demand_unloading.expected(
            bus.route, s, bus.time)
                         for s in range(bus.stop,
                                        self.experiment.routes[bus.route]))
        if transfers := self.experiment.get_transfers(bus.route, bus.stop):
            rate -= sum(t.rate for t in transfers)
        rate_pct = rate / total_rate
        n = sum(self.rng.uniform() < rate_pct for _ in range(bus.passengers))
        t = sum(self.experiment.time_unloading(bus.passengers - i)
                for i in range(n))
        bus.passengers -= n
        event = Event(bus.time, t, 'unload', bus.route, bus.stop,
                      bus.id, bus.passengers)
        bus.time += t
        if transfers:
            bus.state = 'transfer'
            bus.transfers = deque(transfers)
        else:
            bus.state = 'wait'
        return event

    def generate_event_transfer(self, bus):
        """Generates a transfer event."""
        if getattr(bus, 'transfers', None) is None:
            bus.transfers = deque(self.experiment.get_transfers(
                bus.route, bus.stop))
        transfer = bus.transfers.popleft()
        rate_pct = transfer.rate / (sum(
            self.experiment.demand_unloading.expected(bus.route, s, bus.time)
            for s in range(bus.stop+1,
                           self.experiment.routes[bus.route])) +
                                    transfer.rate)
        n = self.rng.binomial(bus.passengers, rate_pct)
        if n:
            t = (sum(self.experiment.time_unloading(bus.passengers - i)
                     for i in range(n)) + transfer.waiting *
                 (bus.time - transfer.last_time) / (transfer.waiting + n))
        else:
            t = 0
        bus.passengers -= n
        transfer.waiting += n
        transfer.last_time = bus.time + t
        event = Event(bus.time, t, 'transfer', bus.route, bus.stop,
                      bus.id, bus.passengers, transfer.waiting)
        bus.time += t
        if not bus.transfers:
            bus.transfers = None
            bus.state = 'wait'
        return event


def simulate(experiment, rng=None):
    """Convenience method to create a new trial object and call
    simulate() on it."""
    return Trial(experiment, rng).simulate()
