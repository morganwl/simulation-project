"""Main functions."""

import argparse
import csv
from collections import defaultdict
import os

import numpy as np

from .trial import simulate
from .experiments import Experiment, get_builtin_experiments

SPEED = 20/60


class Defaults:
    """Default values."""
    # pylint: disable=too-few-public-methods
    # this is a static data object
    numtrials = 1000
    batchsize = 20
    experiment = 'two-stop'
    output = os.path.join('results', '{checksum}.csv')
    params_cache = os.path.join('results', 'params_cache.txt')
    experiments = get_builtin_experiments()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__name__)
    parser.add_argument('--numtrials', '-n',
                        help='number of discrete trials to run',
                        type=int, default=Defaults.numtrials)
    parser.add_argument('--experiment', '-x',
                        help='predefined experiment profile to use',
                        default=Defaults.experiment,
                        type=Defaults.experiments.get)
    parser.add_argument('--output', '-o',
                        help='file to append results to',
                        default=Defaults.output)
    parser.add_argument('--batchsize', '-b',
                        help='number of trials to perform in a single batch',
                        default=Defaults.batchsize)
    parser.add_argument('--params_cache', default=Defaults.params_cache)
    return parser.parse_args()


def measure(events, headers):
    """Returns an array of variables, measured from a list of events."""
    buses = defaultdict(int)
    stops = {}
    handlers = [rv_handlers[h] for h in headers]
    rv = {h: 0 for h in headers}
    for e in events:
        busid = (e.route, e.busid)
        buses[busid] += e.passengers
        for h in handlers:
            h(e, rv, buses, stops)
    for h in rv:
        if h != 'total-passengers' and rv['total-passengers'] > 0:
            rv[h] /= rv['total-passengers']
    return np.fromiter((rv[h] for h in headers), dtype=np.float64,
                       count=len(headers))


def measure_waiting(event, rv, *_):
    """Updates the measurement of waiting rv based on an event."""
    rv['waiting-time'] += event.waiting * event.dur


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
    'waiting-time': measure_waiting,
    'loading-time': measure_loading,
    'moving-time': measure_moving,
    'holding-time': measure_holding,
    'total-passengers': measure_passengers,
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


def update_params_cache(experiment, params_cache):
    """Update the params_cache with the current experiment parameters
    and their checksum."""
    checksum = experiment.checksum()
    if os.path.exists(params_cache):
        with open(params_cache, encoding='utf8') as f:
            for line in f:
                if line.startswith(checksum):
                    return
    with open(params_cache, 'a', encoding='utf8') as f:
        f.write(f'{checksum}: ')
        f.write(str(experiment))
        f.write('\n')


def main(experiment, numtrials, output, batchsize=40, params_cache=None):
    """Performs trials and appends the results for each to an output csv."""
    rng = np.random.default_rng()
    if params_cache is not None:
        update_params_cache(experiment, params_cache)
    print(f'{"var":6s}', end='')
    for h in experiment.headers:
        print(f'{h:>20s}', end='')
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
        for m, ci in zip(means, intervals):
            print(f'{m:8.2f} (+/-{ci[1]-ci[0]:6.2f})', end='')
        print(end='\r')
    var = np.var(trials[:i], axis=0)
    print()
    print(f'{"var":6s}', end='')
    for v in var:
        print(f'{v:20.3f}', end='')
    print()


def confidence_interval(trials, rng, confidence=.95):
    """Returns a (min,max) confidence interval for each column in trials."""
    # NB: numpy percentile inputs percentage as n in [0,100].
    header = 100 * ((1 - confidence) / 2)
    means = np.empty((1000, trials.shape[1]), dtype=np.float64)
    for i in range(means.shape[0]):
        means[i] = np.mean(rng.choice(trials, trials.shape[0]), axis=0)
    intervals = np.column_stack([
        np.percentile(means, header, axis=0),
        np.percentile(means, 100-header, axis=0)])
    return intervals


def cli_entry():
    """Entry point for command line script."""
    options = parse_args()
    options.output = options.output.format(
        checksum=options.experiment.checksum())
    main(options.experiment, options.numtrials, options.output,
         batchsize=options.batchsize, params_cache=options.params_cache)
