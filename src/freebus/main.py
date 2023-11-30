"""Main functions."""

import argparse
import csv
from collections import namedtuple, defaultdict
import os

import numpy as np

from .trial import simulate

SPEED = 20/60


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

def measure_passengers(event, rv, *_):
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

def main(experiment, numtrials, output, batchsize=40):
    """Performs trials and appends the results for each to an output csv."""
    rng = np.random.default_rng()
    print(f'{"var":6s}', end='')
    for h in experiment.headers:
        print(f'{h:>18s}', end='')
    print()
    trials = np.empty((numtrials, len(experiment.headers)), dtype=np.float64)
    i = 0
    while i < numtrials:
        if numtrials - i < batchsize:
            batchsize = numtrials - i
        batch = simulate_batch(experiment, batchsize)
        write_batch(batch, experiment.headers, output)
        trials[i:i+batchsize] = batch
        i += batchsize
        means = np.mean(trials[:i], axis=0)
        intervals = confidence_interval(trials[:i], rng)
        print(f'{"mean":6s}', end='')
        for m,ci in zip(means, intervals):
            print(f'{m:6.3f} ({ci[0]:4.2f},{ci[1]:4.2f})', end='')
        print(end='\r')
    var = np.var(trials[:i], axis=0)
    print()
    print(f'{"var":6s}', end='')
    for v in var:
        print(f'{v:18f}', end='')
    print()

def confidence_interval(trials, rng, confidence=.95):
    """Returns a (min,max) confidence interval for each column in trials."""
    header = (1 - confidence) / 2
    means = np.empty((1000, trials.shape[1]), dtype=np.float64)
    for i in range(means.shape[0]):
        means[i] = np.mean(rng.choice(trials, trials.shape[0]), axis=0)
    intervals = np.column_stack([
        np.percentile(means, header, axis=0),
        np.percentile(means, 1-header, axis=0)])
    return intervals

def cli_entry():
    """Entry point for command line script."""
    options = parse_args()
    main(options.experiment, options.numtrials, options.output,
            batchsize=options.batchsize)
