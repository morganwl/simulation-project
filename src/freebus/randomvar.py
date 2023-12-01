"""Factory functions for random variable generators."""

import numpy as np

class Fixed:
    """A fixed variable returns the same value for any number of
    inputs."""
    def __init__(self, val):
        self.val = np.array(val)
        self.dim = len(self.val.shape)

    def __call__(self, *args):
        return self.val[tuple(args[:self.dim])]

    def __repr__(self):
        return f'{type(self).__name__}({self.val})'

class FixedAlternating:
    """Returns alternating fixed values."""
    def __init__(self, val):
        self.val = np.array(val)
        self.dim = len(self.val.shape) - 1

    def __call__(self, *args):
        values = self.val[tuple(args[:self.dim])]
        self.val[tuple(args[:self.dim])] = np.roll(values, 1)
        return values[-1]

    def __repr__(self):
        return f'{type(self).__name__}({np.array_str(self.val)})'.replace('\n', '')
