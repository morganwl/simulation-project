"""Generate the events for an individual trial."""

from dataclasses import dataclass
from heapq import heappush, heappop

import numpy as np

from .types import Event

SPEED = 20/60

@dataclass
class Stop:
    """State structure for a single bus stop."""
    last_load: float = 0
    remaining: int = 0

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
        f'({self.route}, {self.stop}, {self.time}, id={self.id}, state={self.state})')


    def __lt__(self, other):
        return self.time < other.time

class Trial:
    """Object to manage state for a single trial."""

    def __init__(self, experiment, rng=None):
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
        if bus.state == 'load':
            return self.generate_event_load(bus)
        if bus.state == 'depart':
            return self.generate_event_depart(bus)
        raise RuntimeError('Unknown state.')

    def generate_event_depart(self, bus):
        """Generate a depart event."""
        t = (self.experiment.distance[bus.route][bus.stop] / SPEED /
                self.experiment.traffic(bus.route, bus.stop, bus.time))
        event = Event(bus.time, t, 'depart', bus.route, bus.stop, bus.id, 0)
        bus.time += t
        bus.stop += 1
        if bus.stop < self.experiment.routes[bus.route]:
            bus.state = 'unload'
        else:
            bus.active = False
        return event

    def generate_event_load(self, bus):
        """Generate a load event."""
        delta = bus.time - self.stops[bus.route][bus.stop].last_load
        n = self.experiment.demand_loading(bus.route, bus.stop, bus.time, scale=delta)
        t = sum(self.experiment.time_loading(bus.passengers - i)
                for i in range(n))
        event = Event(bus.time, t, 'load', bus.route, bus.stop, bus.id, n)
        self.stops[bus.route][bus.stop].last_load = bus.time
        bus.time += t
        bus.passengers += n
        if n == 0:
            bus.state = 'depart'
        return event

    def generate_event_unload(self, bus):
        """Generates an unload event."""
        demand_pct = (self.experiment.demand_unloading.expected(bus.route, bus.stop, bus.time)
                / sum(self.experiment.demand_unloading.expected(bus.route, s, bus.time)
                    for s in range(bus.stop, self.experiment.routes[bus.route])))
        n = sum(self.rng.uniform() < demand_pct for _ in range(bus.passengers))
        t = sum(self.experiment.time_unloading(bus.passengers - i)
                for i in range(n))
        event = Event(bus.time, t, 'unload', bus.route, bus.stop, bus.id, -n)
        bus.time += t
        bus.passengers -= n
        bus.state = 'load'
        return event

def simulate(experiment, rng=None):
    """Convenience method to create a new trial object and call
    simulate() on it."""
    return Trial(experiment, rng).simulate()
