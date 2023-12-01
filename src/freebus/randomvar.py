"""Factory functions for random variable generators."""

import numpy as np

class Fixed:
    """A fixed variable returns the same value for any number of
    inputs."""
    def __init__(self, val):
        self.val = np.array(val)
        self.dim = len(self.val.shape)

    def __call__(self, *args, n=None):
        result = self.val[tuple(args[:self.dim])]
        if n is not None:
            return np.ones(n) * result
        return result

    def __repr__(self):
        return f'{type(self).__name__}({self.val})'

class FixedAlternating:
    """Returns alternating fixed values."""
    def __init__(self, val):
        self.val = np.array(val)
        self.dim = len(self.val.shape) - 1

    def __call__(self, *args, n=None):
        if n is not None:
            return np.fromiter((self(*args) for _ in range(n)), dtype=np.float64)
        values = self.val[tuple(args[:self.dim])]
        self.val[tuple(args[:self.dim])] = np.roll(values, 1)
        return values[-1]

    def __repr__(self):
        return f'{type(self).__name__}({np.array_str(self.val)})'.replace('\n', '')

class Pois:
    """Returns a poisson random variable."""
    def __init__(self, mean):
        self.mean = np.array(mean)
        self.dim = len(self.mean.shape)
        self.rng = np.random.default_rng()

    def __call__(self, *args, n=None):
        mean = self.mean[tuple(args[:self.dim])]
        try:
            return self.rng.poisson(mean(*args[self.dim:], n=n))
        except TypeError:
            return self.rng.poisson(mean, size=n)

    def __repr__(self):
        return f'{type(self).__name__}({np.array_str(self.mean)})'.replace('\n', '')
