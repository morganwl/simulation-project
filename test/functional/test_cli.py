"""End-to-end tests of the freebus command line interface."""

import csv
import sys

import freebus as fb

def test_appends_csv(monkeypatch, tmpdir):
    """Tests that an entry is written to csv for each of the specified
    trials."""
    csv_path = tmpdir / 'experiments.csv'
    monkeypatch.setattr(sys, 'argv', ['--numtrials', '5',
        '--experiment', 'simple', '--output', str(csv_path)])
    fb.main.cli_entry()
    with open(csv_path, newline='', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
    assert len(rows) == 6
    fb.main.cli_entry()
    with open(csv_path, newline='', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
    assert len(rows) == 11
