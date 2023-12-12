"""Calculate the average runtime to perform a single trial."""

import sys
import timeit

import freebus


def get_experiment():
    """Returns the b35 builtin experiment."""
    experiments = freebus.experiments.get_builtin_experiments()
    return experiments['b35-short']


def stmt(experiment):
    """Simulates a batch of size 1."""
    freebus.main.simulate_batch(experiment, 1)


def main(n=10):
    """Print the average time to simulate a batch of size 1."""
    total = timeit.timeit('stmt(experiment)',
                          setup=('import freebus;'
                                 'from __main__ import get_experiment, stmt;'
                                 'experiment = get_experiment()'),
                          number=n)
    print(f'{total / n:4.4f}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()
