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
    def __init__(self, route, stop, time, passengers=0, state='unload'):
        self.state = state
        self.route = route
        self.stop = stop
        self.time = time
        self.id = time
        self.passengers = passengers
        self.active = True

    def __repr__(self):
        return (f'{type(self).__name__}'
                f'({self.route}, {self.stop}, {self.time}, '
                f'id={self.id}, state={self.state})')

    def __lt__(self, other):
        return self.time < other.time


class Trial:
    """Object to manage state for a single trial."""

    def __init__(self, experiment, rng=None, uniforms=None):
        experiment.reset(uniforms)
        self.experiment = experiment
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.stops = [[Stop() for _ in range(r)] for r in experiment.routes]

    def simulate(self):
        """Returns a list of simulated events."""
        events = []
        # queue = []
        routes = []
        transfers = {(t.fr_route, t.fr_stop) for t
                     in self.experiment.transfers}
        transfers.update({(t.to_route, t.to_stop) for t
                          in self.experiment.transfers})
        for i, route in enumerate(self.experiment.schedule):
            buses = []
            for bus in route:
                heappush(buses, Bus(i, 0, bus))
            heappush(routes, (buses[0], buses))
        while routes:
            _, buses = routes[0]
            while buses:
                bus = buses[0]
                while bus.active:
                    event = self.generate_event(bus)
                    events.append(event)
                    if bus.state in ['wait', 'transfer_wait']:
                        break
                if bus.active:
                    heapreplace(buses, bus)
                else:
                    heappop(buses)
                if (bus.route, bus.stop) in transfers:
                    break
            if buses:
                heapreplace(routes, (buses[0], buses))
            else:
                heappop(routes)
        return events

    def generate_event(self, bus):
        """Generates an event based on the current state of bus, and updates
        states."""
        if bus.state == 'unload':
            return self.generate_event_unload(bus)
        if bus.state == 'transfer_wait':
            return self.generate_event_transfer_wait(bus)
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

        n, t = self.experiment.demand_loading.arrivals(stop.last_load,
                                                       bus.time,
                                                       bus.route,
                                                       bus.stop)
        t += delta * stop.waiting
        # n = self.experiment.demand_loading(bus.route, bus.stop,
        #                                    bus.time, scale=delta)
        # t = (self.experiment.demand_loading.sum_arrivals(n, delta,
        #                                                  time=bus.time)
        #      + delta * stop.waiting)
        stop.last_load = bus.time

        for transfer in self.experiment.get_transfers_to(bus.route,
                                                         bus.stop):
            if transfer.waiting:
                t += transfer.waiting * (bus.time - transfer.last_time)
                n += transfer.waiting
                transfer.waiting = 0

        if n > 0:
            bus.state = 'load'
            stop.waiting += n
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
        p = self.experiment.unload_pct[bus.route][bus.stop]
        if transfers := self.experiment.get_transfers(bus.route, bus.stop):
            p -= sum(t.p for t in transfers)
        n = self.rng.binomial(bus.passengers, p)
        t = sum(self.experiment.time_unloading(bus.passengers - i)
                for i in range(n))
        bus.passengers -= n
        event = Event(bus.time, t, 'unload', bus.route, bus.stop,
                      bus.id, bus.passengers)
        bus.time += t
        if transfers:
            bus.state = 'transfer_wait'
            bus.transfers = deque(transfers)
        else:
            bus.state = 'wait'
        return event

    def generate_event_transfer_wait(self, bus):
        """Generates an event tracking the wait time of passengers from
        earlier transfers."""
        transfers = self.experiment.get_transfers(bus.route, bus.stop)
        t = sum(bus.time - transfer.last_time
                for transfer in transfers)
        n = sum(transfer.waiting
                for transfer in transfers)
        for transfer in transfers:
            transfer.last_time = bus.time
        bus.state = 'transfer'
        return Event(bus.time, t, 'transfer_wait', bus.route, bus.stop, bus.id,
                     bus.passengers, n)

    def generate_event_transfer(self, bus):
        """Generates a transfer event."""
        if getattr(bus, 'transfers', None) is None:
            bus.transfers = deque(self.experiment.get_transfers(
                bus.route, bus.stop))
        transfer = bus.transfers.popleft()
        n = self.rng.binomial(bus.passengers, transfer.p)
        if n:
            t = (sum(self.experiment.time_unloading(bus.passengers - i)
                     for i in range(n)))
        else:
            t = 0
        bus.passengers -= n
        transfer.waiting += n
        transfer.last_time = bus.time
        event = Event(bus.time, t, 'transfer', bus.route, bus.stop,
                      bus.id, bus.passengers, transfer.waiting)
        bus.time += t
        if not bus.transfers:
            bus.transfers = None
            bus.state = 'wait'
        return event


def simulate(experiment, rng=None, uniforms=None):
    """Convenience method to create a new trial object and call
    simulate() on it."""
    return Trial(experiment, rng, uniforms).simulate()
