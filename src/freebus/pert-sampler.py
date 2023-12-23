"""Performs trials over a range of loading times."""

from pathlib import Path

import numpy as np

from .main import confidence_interval, write_batch, simulate_batch, \
    capacity_scale
from .experiments import get_builtin_experiments
from .randomvar import Pert


def main(experiment, output, points=12, min_mean=3, max_mean=48,
         tol=.25, batchsize=3):
    results = []
    means = np.ones(points)
    perror = np.ones(points)
    active = perror == 0
    active[0], active[points // 2], active[-1] = True, True, True
    perts = np.ones((points, 3))
    perts[:, 1] = np.round(np.linspace(min_mean, max_mean, points))
    perts[:, 0] = np.ceil(perts[:, 1] // 8)
    perts[:, 2] = 120
    perts = perts / 60
    while True:
        while (perror > tol).all():
            for pert, run in zip(perts, active):
                if not run:
                    continue
                experiment.time_loading = Pert(*pert, lamb=5,
                                               scale=capacity_scale)
                batch = simulate_batch(experiment, batchsize)
                write_batch(batch, experiment.headers, output)
                results.append(batch)
                means, perror = process(results, perts, experiment.headers)
                print_status(results, means, perror)
        if np.sum(active) == points:
            tol = tol / 2
            active[:] = False
        active[int(np.round(np.linspace(
            0, points, np.sum(active) + 2)))] = True
    print()


def process(results, perts, headers):
    results = np.array(results)
    pert_index = headers.index('pert-mean')
    time_index = [headers.index[h] for h in
                  ['waiting-time',
                   'loading-time',
                   'moving-time',
                   'holding-time']]
    means = []
    perror = []
    for pert in perts:
        rows = results[:results[pert_index] == pert[1]]
        sums = np.sum(rows[:,time_index])
        means.append(np.mean(sums))
        ci = confidence_interval(sums)
        perror.append((ci[1] - ci[0]) / means[-1])
    return np.array(means), np.array(ci)


def print_status(results, means, perror):
    buffer = [f'{len(results):5d}'] + [
        f'{m:6.2f} ({pe*100}:5.1f)'
        for m, pe in zip(means, perror)]
    print('  '.join(buffer), end='\r')


def cli_entry():
    experiment = get_builtin_experiments()['brooklyn']
    output = Path('results' / 'brooklyn.csv')
    main(experiment, output)
