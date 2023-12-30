"""Main functions."""

import argparse
import csv
import logging
import os

import numpy as np

from .trial import simulate
from .experiments import get_builtin_experiments, capacity_scale
from .measure import measure
from .randomvar import Pert

SPEED = 20/60


class Defaults:
    """Default values."""
    # pylint: disable=too-few-public-methods
    # this is a static data object
    numtrials = 10
    batchsize = 10
    experiment = 'b35-short'
    output = os.path.join('results', '{name}_{checksum}.csv')
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
                        choices=Defaults.experiments)
    parser.add_argument('--pert', '-p',
                        help='parameters for time loading pert distribution',
                        type=seconds)
    parser.add_argument('--output', '-o',
                        help='file to append results to',
                        default=Defaults.output)
    parser.add_argument('--batchsize', '-b',
                        help='number of trials to perform in a single batch',
                        default=Defaults.batchsize)
    parser.add_argument('--params_cache', default=Defaults.params_cache)
    return parser.parse_args()


def seconds(s):
    """Parses a string containing a list of seconds and returns a list
    of floats."""
    return [float(f)/60 for f in s.split(',')]


def simulate_batch(experiment, batch_size, antithetic=False, rng=None, seed=None):
    """Return an array of results for a batch of trials."""
    if rng is None:
        rng = np.random.default_rng(seed)
    if antithetic:
        if batch_size % 4:
            logging.warning((
                'Antithetic variables require a batch size divisible by 4.\n'
                '  %d changed to %d.'), batch_size,
                            batch_size - batch_size % 4)
        batch_size = batch_size // 4
        batch = np.empty((batch_size, len(experiment.headers)),
                         dtype=np.float64)
        for i in range(batch_size):
            group = np.empty((4, len(experiment.headers)), dtype=np.float64)
            j = 0
            u, v = rng.uniform(), rng.uniform()
            for p in [u, 1 - u]:
                for r in [v, 1 - v]:
                    group[j] = measure(simulate(experiment, uniforms=[p, r]),
                                       experiment.headers, experiment.traffic,
                                       experiment.time_loading)
                    j += 1
            batch[i] = np.mean(group, axis=0)
    else:
        batch = np.empty((batch_size, len(experiment.headers)),
                         dtype=np.float64)
        for i in range(batch_size):
            batch[i] = measure(simulate(experiment), experiment.headers,
                               experiment.traffic, experiment.time_loading)
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


def confidence_interval(trials, rng=None, confidence=.95, quantile=.5):
    """Returns a (min,max) confidence interval for each column in trials."""
    # NB: numpy percentile inputs percentage as n in [0,100].
    if rng is None:
        rng = np.random.default_rng()
    header = 100 * ((1 - confidence) / 2)
    means = np.empty((1000, trials.shape[1]), dtype=np.float64)
    for i in range(means.shape[0]):
        means[i] = np.mean(rng.choice(trials, trials.shape[0]), axis=0)
    intervals = np.column_stack([
        np.percentile(means, header, axis=0),
        np.percentile(means, 100-header, axis=0)])
    return intervals


def main(experiment, numtrials, output, batchsize=40, pert=None, params_cache=None):
    """Performs trials and appends the results for each to an output csv."""
    rng = np.random.default_rng()
    if params_cache is not None:
        update_params_cache(experiment, params_cache)
    if pert is not None:
        experiment.time_loading = Pert(*pert, lamb=3, scale=capacity_scale)
        experiment.gather_pert = True
    print(f'{"var":6s}', end='')
    for h in experiment.headers[:5]:
        print(f'{h:>18s}', end='')
    print()
    trials = np.empty((numtrials, len(experiment.headers)), dtype=np.float64)
    i = 0
    while i < numtrials:
        if numtrials - i < batchsize:
            batchsize = numtrials - i
        batch = simulate_batch(experiment, batchsize, antithetic=True)
        write_batch(batch, experiment.headers, output)
        trials[i:i+batchsize] = batch
        i += batchsize
        means = np.mean(trials[:i], axis=0)
        intervals = confidence_interval(trials[:i], rng)
        print(f'{"mean":6s}', end='')
        for m, ci in zip(means, intervals[:5]):
            print(f'{m:9.1f} +/-{ci[1]-ci[0]:5.1f}', end='')
        print(end='\r')
    var = np.var(trials[:i], axis=0)
    print()
    print(f'{"var":6s}', end='')
    for v in var[:5]:
        print(f'{v:18.3f}', end='')
    print()


def cli_entry():
    """Entry point for command line script."""
    options = parse_args()
    experiment = Defaults.experiments.get(options.experiment)
    options.output = options.output.format(
        name=options.experiment, checksum=experiment.checksum())
    print(options.pert)
    main(experiment, options.numtrials, options.output,
         batchsize=options.batchsize, pert=options.pert,
         params_cache=options.params_cache)
