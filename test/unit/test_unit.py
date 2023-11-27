"""Collected unit tests."""

import csv

import numpy as np

import freebus as fb

def test_write_batch(tmpdir):
    """Tests appending a batch of trials to a csv file."""
    output = tmpdir / 'test_output.csv'
    headers = ['first', 'second', 'third']
    expected = [headers[:]]
    batch = np.array([[1., 2., 3.], [4., 5., 6.]])
    expected.extend(batch.tolist())
    fb.main.write_batch(batch, headers, output)
    batch = np.array([[7., 8., 9.], [10., 11., 12.]])
    expected.extend(batch.tolist())
    fb.main.write_batch(batch, headers, output)
    print(expected)
    with open(output, newline='', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile)
        result = [next(reader)]
        result.extend([[float(v) for v in row] for row in reader])
        assert expected == result
