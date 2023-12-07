"""Plot results."""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from .main import Defaults


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__name__)
    parser.add_argument('input')
    parser.add_argument('--params_cache', default=Defaults.params_cache)
    return parser.parse_args()


def main(source):
    data = np.loadtxt(source, skiprows=1, delimiter=',')
    print(data.shape)
    print(np.sum(data[:, :3], axis=1).shape)
    plt.hist(np.sum(data[:, :3], axis=1), density=True)
    plt.show()


def cli_entry():
    """Entry point for command line script."""
    parsed_args = parse_args()
    main(parsed_args.input)
