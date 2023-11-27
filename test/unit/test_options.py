"""Test handling of command line options."""

import sys

import freebus as fb

def test_experiment_long(monkeypatch):
    """Calling with the --experiment option should assign an experiment
    object to the options.experiment property."""
    monkeypatch.setattr(sys, 'argv', [__name__, '--experiment', 'simple'])
    options = fb.main.parse_args()
    assert isinstance(options.experiment, fb.main.Experiment)

def test_experiment_unit(monkeypatch):
    """Calling with the --experiment option should search find the
    experiment object with the corresponding key in the experiments
    list."""
    monkeypatch.setattr(fb.main.Defaults, 'experiments',
            {'a': 'A', 'b': 'B'})
    monkeypatch.setattr(sys, 'argv', [__name__, '--experiment', 'b'])
    options = fb.main.parse_args()
    assert options.experiment == 'B'
