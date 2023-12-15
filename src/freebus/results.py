"""Plot results."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .main import Defaults, confidence_interval

COLS = 2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__name__)
    parser.add_argument('input', nargs='+', type=Path)
    parser.add_argument('--params_cache', default=Defaults.params_cache)
    return parser.parse_args()


def plot_travel_time(dataset, cols, name, ax):
    """Plot a histogram for total travel times from a single dataset."""
    travel_time = np.sum(dataset[:, [cols['waiting-time'],
                                     cols['loading-time'],
                                     cols['moving-time'],
                                     cols['holding-time']]],
                         axis=1)
    confidence = confidence_interval(np.array([travel_time]).transpose())[0]
    confidence = confidence[1] - confidence[0]
    ax.hist(travel_time, density=True)
    ax.title.set_text(f'{name}\n+/-{confidence:.3f} out of {len(travel_time)}')


def plot_pph(dataset, cols, name, ax):
    """Plots passengers per hour from a single dataset."""
    ax.bar(range(24), np.mean(dataset[:, [cols[f'passengers-{i}'] for i in range(24)]],
           axis=0))
    ax.title.set_text(f'{name}')


def plot_travel_times(datasets):
    """Plot the total travel times for one or more datasets
    side-by-side."""
    fig, subplots = plt.subplots((len(datasets) + COLS - 1) // COLS, COLS,
                                 squeeze=True, sharey=True, sharex=True)
    fig.suptitle('Total Travel Time')
    for ((ds, cols, name), ax) in zip(datasets, subplots):
        plot_travel_time(ds, cols, name, ax)
    plt.show()


def plot_passengers_per_hour(datasets):
    """Plot mean passengers per hour for one or more datasets
    side-by-side."""
    fig, subplots = plt.subplots((len(datasets) + COLS - 1) // COLS, COLS,
                                 squeeze=True, sharey=True, sharex=True)
    fig.suptitle('Passengers per hour')
    for ((ds, cols, name), ax) in zip(datasets, subplots):
        plot_pph(ds, cols, name, ax)
    plt.show()


def main(sources):
    """Generate plots of results."""
    datasets = []
    for source in sources:
        with open(source, encoding='utf8', newline='') as f:
            reader = csv.reader(f)
            cols = {c: i for i, c in enumerate(next(reader))}
            data = np.loadtxt(f, delimiter=',')
        datasets.append((data, cols, source.stem))
    plot_travel_times(datasets)
    plot_passengers_per_hour(datasets)


def cli_entry():
    """Entry point for command line script."""
    parsed_args = parse_args()
    main(parsed_args.input)
