"""Plot results."""

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np

from .main import Defaults, confidence_interval
from .experiments import get_builtin_experiments
from .randomvar import Pert

COLS = 2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__name__)
    parser.add_argument('input', nargs='+')
    parser.add_argument('--params_cache', default=Defaults.params_cache)
    parser.add_argument('--name', '-n')
    parser.add_argument('--dir', '-d', type=Path,
                        default='figures')
    return parser.parse_args()


class Output:
    """Global state for outputting figures."""
    output_dir = Path('figures')
    generated_figures = []
    fmt = 'png'
    name = None

    @classmethod
    def figure(cls, figure, dataset, figname):
        """Output a figure using configured output method."""
        dsname = cls.name if cls.name else dataset
        name = f'{dsname}_{figname}.{cls.fmt}'
        figure.tight_layout()
        figure.savefig(cls.output_dir / name)
        cls.generated_figures.append(name)

    @classmethod
    def set_output(cls, output):
        """Set the output directory."""
        cls.output_dir = Path(output)

    @classmethod
    def set_name(cls, name):
        """Set the base name for generated figures."""
        cls.name = name

    @classmethod
    def from_namespace(cls, options):
        """Configure output based on a namespace object."""
        cls.output_dir = options.dir
        cls.name = options.name

    @classmethod
    def html_report(cls):
        """Generates a simple html report."""
        name = 'report' if cls.name is None else cls.name
        with open(cls.output_dir / (name + '.html'), 'wt',
                  encoding='utf8') as f:
            f.write(f'<html><head><title>{name}</title></head>')
            f.write('<body>\n')
            for fig in cls.generated_figures:
                f.write(f'<img src={fig}>\n')
            f.write('</body></html>')


def plot_tvl(dataset, cols, name, ax, quantile=.5):
    loading = np.unique(dataset[:, cols['pert-mean']])
    groups = (dataset[:, cols['pert-mean']] == u
              for u in loading)
    groups = (dataset[g] for g in groups)
    sums = (np.sum(g[:, [cols['waiting-time'],
                         cols['loading-time'],
                         cols['moving-time'],
                         cols['holding-time']]],
                   axis=1) for g in groups)
    sums = list(sums)
    travel_time = np.fromiter((np.quantile(s, quantile) for s in sums),
                              dtype=np.float64)
    confidence = [confidence_interval(np.array([s]).transpose(),
                                      quantile=quantile)[0]
                  for s in sums]
    upper_confidence = np.array([c1 for c0, c1 in confidence])
    lower_confidence = np.array([c0 for c0, c1 in confidence])
    seconds = loading * 60
    ax.plot(seconds, travel_time)
    ax.plot(seconds, upper_confidence, alpha=0.5)
    ax.plot(seconds, lower_confidence, alpha=0.5)
    plot_estimate_line(dataset, cols, ax, loading[3:], travel_time[3:])
    ax.title.set_text(f'{name}')


def plot_estimate_line(dataset, cols, ax, loading, travel_time):
    base_group = dataset[dataset[:, cols['pert-mean']] == loading[0]]
    base_loading = np.mean(base_group[:, cols['loading-time']]) / 3
    loading = loading * 60
    estimate_travel = (travel_time[0] + (loading / loading[0] - 1) *
                       base_loading)
    ax.plot(loading, estimate_travel)


def plot_travel_vs_loading(datasets):
    fig, subplots = plt.subplots((len(datasets) + COLS - 1) // COLS, COLS,
                                 sharey=True, sharex=True)
    fig.suptitle('Travel vs loading mean')
    for ((ds, cols, name), ax) in zip(datasets, itertools.chain(*subplots)):
        plot_tvl(ds, cols, name, ax)
    Output.figure(fig, datasets[0][2], 'tvl')


def plot_loading_distribution():
    fig = plt.figure()
    fig.suptitle('Sample loading distribution')
    pert = Pert(1/60, 12/60, 120/60, lamb=6)
    plt.hist([pert() for _ in range(50000)], bins=20)
    pert = Pert(1/60, 3/60, 120/60, lamb=6)
    plt.hist([pert() for _ in range(50000)], bins=20)
    Output.figure(fig, 'pert', 'pert')


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
    ax.bar(range(24), np.mean(dataset[:, [cols[f'passengers-{i}']
                                          for i in range(24)]],
                              axis=0))
    ax.title.set_text(f'{name}')


def plot_travel_times(datasets):
    """Plot the total travel times for one or more datasets
    side-by-side."""
    fig, subplots = plt.subplots((len(datasets) + COLS - 1) // COLS, COLS,
                                 sharey=True, sharex=True)
    fig.suptitle('Total Travel Time')
    for ((ds, cols, name), ax) in zip(datasets, itertools.chain(*subplots)):
        plot_travel_time(ds, cols, name, ax)
    Output.figure(fig, datasets[0][2], 'travel')


def plot_passengers_per_hour(datasets):
    """Plot mean passengers per hour for one or more datasets
    side-by-side."""
    fig, subplots = plt.subplots((len(datasets) + COLS - 1) // COLS, COLS,
                                 sharey=True, sharex=True)
    fig.suptitle('Passengers per hour')
    for ((ds, cols, name), ax) in zip(datasets, itertools.chain(*subplots)):
        plot_pph(ds, cols, name, ax)
    Output.figure(fig, datasets[0][2], 'pph')


def plot_traffic_daily(datasets):
    """Plot a histogram of daily traffic volume."""
    fig, subplots = plt.subplots((len(datasets) + COLS - 1) // COLS, COLS,
                                 sharey=True, sharex=True)
    fig.suptitle('Daily traffic volume')
    for ((ds, cols, name), ax) in zip(datasets, itertools.chain(*subplots)):
        plot_traffic(ds, cols, name, ax)
    Output.figure(fig, datasets[0][2], 'traffic')


def plot_traffic(dataset, cols, name, ax):
    """Plots a histogram of daily traffic volume for one dataset."""
    traffic = dataset[:, cols['traffic-daily']]
    ax.hist(traffic, density=True)
    ax.title.set_text(f'{name}')


def plot_traffic_per_hour(datasets):
    fig, subplots = plt.subplots((len(datasets) + COLS - 1) // COLS, COLS,
                                 sharey=True, sharex=True)
    fig.suptitle('Traffic per hour')
    for ((ds, cols, name), ax) in zip(datasets, itertools.chain(*subplots)):
        plot_tph(ds, cols, name, ax)
    Output.figure(fig, datasets[0][2], 'tph')


def plot_tph(dataset, cols, name, ax):
    """Plots average distribution of traffic per hour for one
    dataset."""
    ax.bar(range(24), np.mean(dataset[:, [cols[f'traffic-{i}']
                                          for i in range(24)]],
                              axis=0))
    ax.title.set_text(f'{name}')


def plot_passengers_daily(datasets):
    fig, subplots = plt.subplots((len(datasets) + COLS - 1) // COLS, COLS,
                                 sharey=True, sharex=True)
    fig.suptitle('Daily passengers')
    for ((ds, cols, name), ax) in zip(datasets, itertools.chain(*subplots)):
        plot_passengers(ds, cols, name, ax)
    Output.figure(fig, datasets[0][2], 'passengers')


def plot_passengers(dataset, cols, name, ax):
    passengers = dataset[:, cols['total-passengers']]
    ax.hist(passengers, density=True)
    ax.title.set_text(f'{name}')


def expand_results(sources):
    builtins = get_builtin_experiments()
    for i, source in enumerate(sources):
        source = str(source)
        if source in builtins:
            filename = f'{source}_{builtins[source].checksum()}.csv'
            sources[i] = Path('results') / filename
    return sources


def expand_source(source):
    builtins = get_builtin_experiments()
    if source in builtins:
        filename = f'{source}_{builtins[source].checksum()}.csv'
        return Path('results') / filename
    return Path(source)


def main(sources):
    """Generate plots of results."""
    datasets = []
    for source in sources:
        filename = expand_source(source)
        with open(filename, encoding='utf8', newline='') as f:
            reader = csv.reader(f)
            cols = {c: i for i, c in enumerate(next(reader))}
            data = np.loadtxt(f, delimiter=',')
        datasets.append((data, cols, source))
    plot_travel_times(datasets)
    plot_passengers_per_hour(datasets)
    plot_passengers_daily(datasets)
    plot_traffic_daily(datasets)
    plot_traffic_per_hour(datasets)
    plot_travel_vs_loading(datasets)
    plot_loading_distribution()
    Output.html_report()


def cli_entry():
    """Entry point for command line script."""
    parsed_args = parse_args()
    Output.from_namespace(parsed_args)
    main(parsed_args.input)
