"""Plot results."""

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

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
        """Generate a simple html report."""
        name = 'report' if cls.name is None else cls.name
        with open(cls.output_dir / (name + '.html'), 'wt',
                  encoding='utf8') as f:
            f.write(f'<html><head><title>{name}</title></head>')
            f.write('<body>\n')
            for fig in cls.generated_figures:
                f.write(f'<img src={fig}>\n')
            f.write('</body></html>')


def plot_tvl(dataset, cols, name, ax, quantile=.5):
    """Scatter plot total travel time vs mean loading time."""
    loading = np.unique(dataset[:, cols['pert-mean']])
    groups = (dataset[:, cols['pert-mean']] == u
              for u in loading)
    groups = (dataset[g] for g in groups)
    sums = (np.sum(g[:, [cols['waiting-time'],
                         cols['loading-time'],
                         cols['moving-time'],
                         cols['holding-time']]],
                   axis=1) for g in groups)
    ax.set_ylim([110, 165])
    seconds = loading * 60
    sums = list(sums)
    for load_time, travel_time in zip(seconds, sums):
        num_trials = len(travel_time)
        x = np.full(num_trials, load_time)
        ax.scatter(x, [np.median(np.random.choice(travel_time,
                                                  num_trials))
                       for _ in range(num_trials)],
                   alpha=3e-2,
                   s=75,
                   c='#1f77b4')
        ax.scatter(load_time, np.median(travel_time),
                   alpha=1e-1,
                   s=125,
                   c='#1f77b4')
    ax.plot([seconds[0], seconds[-1]],
            [np.median(sums[0]), np.median(sums[-1])])


def plot_tvl3(dataset, cols, name, ax, quantile=.5):
    """Plot differences in travel time vs loading time.

    Calculate differences for a random sampling of paired trials.
    """
    minuend = dataset[:, [cols['pert-mean'],
                          cols['waiting-time'],
                          cols['loading-time'],
                          cols['holding-time'],
                          cols['moving-time'],]]
    minuend[:, 0] = minuend[:, 0] * 60
    minuend = minuend[np.random.choice(minuend.shape[0], 50000), :]
    subtrahend = np.random.permutation(minuend)
    difference = minuend - subtrahend
    difference[:, 0] = np.round(difference[:, 0], 3)
    difference[difference[:, 0] < 0, :] = -difference[difference[:, 0] < 0, :]
    # -(
    #     difference[difference[:, 0] < 0, [False] + [True] * 4])
    loading = np.unique(difference[:, 0])
    groups = [
        np.sum(
            difference[np.ix_(difference[:, 0] == lt,
                              np.arange(1, 5))],
            axis=1)
        for lt in loading]
    confidence = [confidence_interval(np.array([g]).transpose(),
                                      quantile=.5)[0]
                  for g in groups]
    ax.plot(loading, [np.mean((g)) for g in groups])
    m0, b0, r0, p0, e0 = scipy.stats.linregress(loading,
                                                [c0 for c0, c1 in
                                                 confidence])
    m1, b1, r1, p1, e1 = scipy.stats.linregress(loading, [c1 for c0, c1
                                                          in
                                                          confidence])
    ax.plot(loading, m0 * loading + b0,
            alpha=.75 * abs(r0) * (1 - p0))
    ax.plot(loading, m1 * loading + b1,
            alpha=.75 * abs(r1) * (1 - p1))
    max_mean = np.mean(groups[-1])
    max_confidence = max(confidence[-1][1] - max_mean,
                         max_mean - confidence[-1][0])
    con_mean = np.mean(groups[10])
    con_con = max(confidence[10][1] - con_mean,
                  con_mean - confidence[10][0])
    ax.title.set_text('change in travel time vs change in loading time\n'
                      f'{np.max(loading)} seconds:{max_mean:5.1f} minutes '
                      f'+/- {max_confidence:.2f} minutes\n'
                      f'{loading[10]} seconds:{con_mean:5.1f} minutes '
                      f'+/- {con_con:.2f} minutes\n')


def plot_tvl2(dataset, cols, name, ax, quantile=.5):
    """Plot total travel time vs mean loading time.

    Plots travel time as a line, as well as ploting upper and lower
    bounds of confidence interval.
    """
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
    travel_time = np.fromiter((np.mean(s) for s in sums),
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
    difference = travel_time[-1] - travel_time[0]
    ax.title.set_text(
        f'{name}\n{difference:.1f} '
        f'(+{upper_confidence[-1] - lower_confidence[0] - difference:.2f}, '
        f'-{difference - lower_confidence[-1] + upper_confidence[0]:.2f})')


def plot_estimate_line(dataset, cols, ax, loading, travel_time):
    """Plot linear estimate of travel time vs mean loading time.

    Line is drawn between mean of shortest and longest observed mean
    loading time.
    """
    base_group = dataset[dataset[:, cols['pert-mean']] == loading[0]]
    base_loading = np.mean(base_group[:, cols['loading-time']]) / 3
    loading = loading * 60
    estimate_travel = (travel_time[0] + (loading / loading[0] - 1) *
                       base_loading)
    ax.plot(loading, estimate_travel)


def plot_loading_times(dataset, cols, name, ax, lamb=5):
    """Plot minimum, median, and maximum loading distributions."""
    loading_times = np.unique(dataset[:,
                                      (cols['pert-min'],
                                       cols['pert-mean'],
                                       cols['pert-max'],)],
                              axis=0)
    for i in [0, len(loading_times) // 2, len(loading_times) - 1]:
        lt = loading_times[i]
        distribution = 60 * Pert(*lt, lamb=lamb)(n=1000)
        plt.hist(distribution, bins=20, density=True,
                 histtype='bar',)


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
    """Plot passengers per hour from a single dataset."""
    ax.bar(range(24), np.mean(dataset[:, [cols[f'passengers-{i}']
                                          for i in range(24)]],
                              axis=0))
    ax.title.set_text(f'{name}')


def plot_traffic(dataset, cols, name, ax):
    """Plot a histogram of daily traffic volume for one dataset."""
    traffic = dataset[:, cols['traffic-daily']]
    ax.hist(traffic, density=True)
    ax.title.set_text(f'{name}')


def plot_tph(dataset, cols, name, ax):
    """Plot average distribution of traffic per hour."""
    ax.bar(range(24), np.mean(dataset[:, [cols[f'traffic-{i}']
                                          for i in range(24)]],
                              axis=0))
    ax.title.set_text(f'{name}')


def plot_passengers(dataset, cols, name, ax):
    """Plot distribution of total passengers."""
    passengers = dataset[:, cols['total-passengers']]
    ax.hist(passengers, density=True)
    ax.title.set_text(f'{name}')


def plot_heavy_traffic_loading(dataset, cols, name, ax,
                               quantile=0.75):
    """Plot the travel time vs loading for days with heavy traffic."""
    heavy_traffic = np.quantile(dataset[:, cols['traffic-daily']],
                                q=quantile)
    dataset = dataset[dataset[:, cols['traffic-daily']] >= heavy_traffic, :]
    loading_times = np.unique(dataset[:, cols['pert-mean']])
    means = np.array([np.mean(
        np.sum(dataset[np.ix_(
            dataset[:, cols['pert-mean']] == lt,
            [cols['waiting-time'],
             cols['loading-time'],
             cols['holding-time'],
             cols['moving-time']])],
               axis=1)) for lt in loading_times])
    ax.plot(60 * loading_times, means)


def plot_heavy_passenger_loading(dataset, cols, name, ax,
                                 quantile=0.9):
    """Plot the travel time vs loading for days with heavy demand."""
    heavy_pass = np.quantile(dataset[:, cols['total-passengers']],
                             q=quantile)
    dataset = dataset[dataset[:, cols['total-passengers']] >= heavy_pass, :]
    loading_times = np.unique(dataset[:, cols['pert-mean']])
    means = np.array([np.mean(
        np.sum(dataset[np.ix_(
            dataset[:, cols['pert-mean']] == lt,
            [cols['waiting-time'],
             cols['loading-time'],
             cols['holding-time'],
             cols['moving-time']])],
               axis=1)) for lt in loading_times])
    ax.plot(60 * loading_times, means)


def plot_buscap_load(dataset, cols, name, ax):
    """Plot upper passenger counts vs loading time."""
    loading_times = np.unique(dataset[:, cols['pert-mean']])
    buscap = np.array([
        np.mean(dataset[dataset[:, cols['pert-mean']] == lt,
                        cols['extreme-load']])
        for lt in loading_times])
    ax.plot(loading_times * 60,
            buscap)


def new_figure(datasets, name, func):
    """Initialize a figure for a plotting function."""
    fig, subplots = plt.subplots((len(datasets) + COLS - 1) // COLS,
                                 COLS if len(datasets) > COLS
                                 else len(datasets),
                                 squeeze=False, sharey=False, sharex=False)
    fig.suptitle(name)
    for ((ds, cols, filename), ax) in zip(datasets,
                                          itertools.chain(*subplots)):
        func(ds, cols, filename, ax)
    Output.figure(fig, datasets[0][2], name.replace(' ', '_'))


def expand_source(source):
    """Match experiment name to results file for latest parameters."""
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
    for func, name in [
            (plot_tvl3, ''),
            (plot_travel_time, 'daily travel time'),
            (plot_passengers, 'daily passenger volume'),
            (plot_pph, 'passengers per hour'),
            (plot_traffic, 'daily traffic volume'),
            (plot_tph, 'traffic per hour'),
            (plot_tvl, 'travel time vs boarding time'),
            (plot_tvl2, 'travel time vs boarding time alt'),
            (plot_loading_times, 'loading time distributions'),
            (plot_heavy_traffic_loading,
             'travel time vs boarding under heavy traffic'),
            (plot_heavy_passenger_loading,
             'travel time vs boarding under heavy passengers'),
            (plot_buscap_load,
             'extreme bus load vs loading time'),
    ]:
        new_figure(datasets, name, func)
    Output.html_report()


def cli_entry():
    """Entry point for command line script."""
    parsed_args = parse_args()
    Output.from_namespace(parsed_args)
    main(parsed_args.input)
