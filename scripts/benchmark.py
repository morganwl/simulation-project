"""Calculate the average runtime to perform a single trial."""

import argparse
import cProfile
from collections import Counter
import numpy as np
import pstats
import timeit

import freebus


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['time', 'profile', 'events'],
                        default='time', nargs='?')
    parser.add_argument('-n', type=int, default=10)
    return parser.parse_args()


def get_experiment():
    """Returns the b35 builtin experiment."""
    experiments = freebus.experiments.get_builtin_experiments()
    return experiments['b35-short']


def stmt(experiment, n=1):
    """Simulates a batch of size 1."""
    freebus.main.simulate_batch(experiment, n)


def events_main(n=10):
    """Print the average count of events generated by a simulation."""
    counter = Counter()
    experiment = get_experiment()
    for _ in range(n):
        trial = freebus.trial.Trial(experiment)
        for e in trial.simulate():
            counter[e.etype] += 1
    for t, c in counter.items():
        print(f'{t + ":" :12s} {c/n:.1f}')
    print(f'{"total:":12s} {counter.total()/n:.1f}')


def profile_main(n=10):
    # pylint: disable=unused-argument
    """Print a profiling report."""
    experiment = get_experiment()
    with cProfile.Profile() as pr:
        stmt(experiment, n)
        stats = pstats.Stats(pr)
    stats.strip_dirs()
    stats.reverse_order()
    stats.sort_stats('cumtime')
    stats.reverse_order()
    stats.print_stats()


def time_main(n=10):
    """Print the average time to simulate a batch of size 1."""
    total = timeit.timeit('stmt(experiment)',
                          setup=('import freebus;'
                                 'from __main__ import get_experiment, stmt;'
                                 'experiment = get_experiment()'),
                          number=n)
    print(f'{total / n:4.4f}')


if __name__ == '__main__':
    options = parse_args()
    match options.command:
        case 'time':
            time_main(options.n)
        case 'profile':
            profile_main(options.n)
        case 'events':
            events_main(options.n)
