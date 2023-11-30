"""Main functions."""

import argparse
import csv
from collections import namedtuple, defaultdict
from dataclasses import dataclass
import os
from heapq import heappush, heappop
from random import random as uniform

import numpy as np

SPEED = 20/60

Event = namedtuple('Event', ['time', 'dur', 'etype', 'route', 'stop', 'busid', 'passengers'])

class Experiment:
    """Object containing experimental parameters."""
    def __init__(self, routes, distance, traffic, demand_loading,
            demand_unloading, time_loading, time_unloading,
            schedule, headers):
        self.routes = routes
        self.distance = distance
        self.traffic = traffic
        self.demand_loading = demand_loading
        self.demand_unloading = demand_unloading
        self.time_loading = time_loading
        self.time_unloading = time_unloading
        self.schedule = schedule
        self.headers = headers


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
        return f'{type(self).__name__}({self.route}, {self.stop}, {self.time}, id={self.id}, state={self.state})'


    def __lt__(self, other):
        return self.time < other.time

class Defaults:
    """Default values."""
    # pylint: disable=too-few-public-methods
    # this is a static data object
    numtrials = 1000
    batchsize = 20
    experiment = None
    experiments = {
            'simple': Experiment(
                [2], [[0, 1]],
                lambda r,s,t: 1,
                lambda r,s,t,d: 1 if t in {0, 10} else 0,
                lambda r,s,t: [0,1][s],
                lambda p: 1,
                lambda p: 1,
                [[0, 10]],
                ['loading-time', 'moving-time', 'holding-time', 'total-passengers']),
            }

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__name__)
    parser.add_argument('--numtrials', '-n', help='number of discrete trials to run',
            type=int, default=Defaults.numtrials)
    parser.add_argument('--experiment', '-x', help='predefined experiment profile to use',
            default=Defaults.experiment, type=Defaults.experiments.get)
    parser.add_argument('--output', '-o', help='file to append results to')
    parser.add_argument('--batchsize', '-b',
            help='number of trials to perform in a single batch',
            default=Defaults.batchsize)
    return parser.parse_args()


def generate_event(bus, stops, experiment):
    """Generates an event based on the current state of bus, and updates
    states."""
    if bus.state == 'unload':
        return generate_event_unload(bus, experiment)
    if bus.state == 'load':
        return generate_event_load(bus, stops, experiment)
    if bus.state == 'depart':
        return generate_event_depart(bus, experiment)


def generate_event_depart(bus, experiment):
    t = (experiment.distance[bus.route][bus.stop] / SPEED /
            experiment.traffic(bus.route, bus.stop, bus.time))
    event = Event(bus.time, t, 'depart', bus.route, bus.stop, bus.id, 0)
    bus.time += t
    bus.stop += 1
    if bus.stop < experiment.routes[bus.route]:
        bus.state = 'unload'
    else:
        bus.active = False
    return event

def generate_event_load(bus, stops, experiment):
    delta = bus.time - stops[bus.route][bus.stop].last_load
    n = experiment.demand_loading(bus.route, bus.stop, bus.time, delta)
    t = sum(experiment.time_loading(bus.passengers - i)
            for i in range(n))
    event = Event(bus.time, t, 'load', bus.route, bus.stop, bus.id, n)
    stops[bus.route][bus.stop].last_load = bus.time
    bus.time += t
    bus.passengers += n
    if n == 0:
        bus.state = 'depart'
    return event


def generate_event_unload(bus, experiment):
    """Generates an unload event."""
    demand_pct = (experiment.demand_unloading(bus.route, bus.stop, bus.time)
            / sum(experiment.demand_unloading(bus.route, s, bus.time)
                for s in range(bus.stop, experiment.routes[bus.route])))
    n = sum(uniform() < demand_pct for _ in range(bus.passengers))
    t = sum(experiment.time_unloading(bus.passengers - i)
            for i in range(n))
    event = Event(bus.time, t, 'unload', bus.route, bus.stop, bus.id, -n)
    bus.time += t
    bus.passengers -= n
    bus.state = 'load'
    return event


def simulate(experiment):
    """Returns a list of simulated events."""
    events = []
    queue = []
    stops = []
    for route in experiment.routes:
        r = []
        for _ in range(route):
            r.append(Stop())
        stops.append(r)
    for i, route in enumerate(experiment.schedule):
        for bus in route:
            heappush(queue, Bus(i, 0, bus))
    while queue:
        bus = heappop(queue)
        event = generate_event(bus, stops, experiment)
        events.append(event)
        if bus.active:
            heappush(queue, bus)
    return events


def measure(events, headers):
    """Returns an array of variables, measured from a list of events."""
    buses = defaultdict(int)
    stops = {}
    handlers = [rv_handlers[h] for h in headers]
    rv = {h:0 for h in headers}
    for e in events:
        busid = (e.route, e.busid)
        buses[busid] += e.passengers
        for h in handlers:
            h(e, rv, buses, stops)
    return np.fromiter((rv[h] for h in headers), dtype=np.float64, count=len(headers))

def measure_loading(event, rv, buses, _):
    """Updates the measurement of loading rv based on an event."""
    if event.etype in ['load', 'unload']:
        rv['loading-time'] += buses[(event.route, event.busid)] * event.dur

def measure_moving(event, rv, buses, _):
    """Updates the measurement of moving rv based on an event."""
    if event.etype == 'depart':
        rv['moving-time'] += buses[(event.route, event.busid)] * event.dur

def measure_holding(event, rv, buses, _):
    """Measures the time spent holding."""
    if event.etype == 'hold':
        rv['holding-time'] += buses[(event.route, event.busid)] * event.dur

def measure_passengers(event, rv, *args):
    """Measures the number of passengers."""
    if event.etype == 'load':
        rv['total-passengers'] += event.passengers

rv_handlers = {
        'loading-time': measure_loading,
        'moving-time': measure_moving,
        'holding-time': measure_holding,
        'total-passengers': measure_passengers
        }

def simulate_batch(experiment, batch_size):
    """Returns an array of results for a batch of trials."""
    batch = np.empty((batch_size, len(experiment.headers)), dtype=np.float64)
    for i in range(batch_size):
        batch[i] = measure(simulate(experiment), experiment.headers)
    return batch


def write_batch(batch, headers, output):
    """Appends a batch to a csv file, creating a new one with headers if
    no such file exists."""
    write_headers = False
    if not os.path.exists(output) or os.path.getsize(output) == 0:
        write_headers = True
    with open(output, 'at', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        if write_headers:
            writer.writerow(headers)
        writer.writerows(batch)


def main(experiment, numtrials, output, batchsize=20):
    """Performs trials and appends the results for each to an output csv."""
    batches = []
    while numtrials:
        if numtrials < batchsize:
            batchsize = numtrials
        batch = simulate_batch(experiment, batchsize)
        write_batch(batch, experiment.headers, output)
        batches.append(batch)
        numtrials -= batchsize

def cli_entry():
    """Entry point for command line script."""
    options = parse_args()
    main(options.experiment, options.numtrials, options.output,
            batchsize=options.batchsize)
