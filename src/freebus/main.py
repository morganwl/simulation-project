"""Main functions."""

import argparse
import csv
import os

import numpy as np

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
                [2], [0, 1],
                lambda r,s,t: 1,
                lambda r,s,t: [1,0][s],
                lambda r,s,t: [0,1][s],
                lambda p: 1,
                lambda p: 1,
                [0, 10],
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

# def estimate(experiment, n):
#     trials = np.fromiter((simulate(experiment) for _ in range(n)), np.float64,
#             n)
#     return np.mean(trials)

# def simulate(experiment):
#     pass

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
    numbatches = np.ceil(numtrials / batchsize)
    for _ in range(numbatches):
        batch = simulate_batch(experiment, batchsize)
        write_batch(batch, experiment.headers, output)
        print(batch)
        batches.append(batch)

def cli_entry():
    """Entry point for command line script."""
    options = parse_args()
    main(options.experiment, options.numtrials, options.output,
            batchsize=options.batchsize)
