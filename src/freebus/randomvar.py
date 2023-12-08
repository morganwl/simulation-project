"""Factory functions for random variable generators."""

import numpy as np


class Fixed:
    """A fixed variable returns the same value for any number of
    inputs."""
    def __init__(self, val):
        self.val = np.array(val)
        self.dim = len(self.val.shape)

    def __call__(self, *args, scale=1, n=None):
        result = self.val[tuple(args[:self.dim])]
        if n is not None:
            return np.ones(n) * result
        return result

    def expected(self, *args):
        """Returns the expected value for any given parameters."""
        return self.val[tuple(args[:self.dim])]

    def distribute_time(self, n, t):
        """Returns the sum of arrival times, given n arrivals over time t."""
        return sum(i * t/(n+1) for i in range(1, n+1))

    def __repr__(self):
        return f'{type(self).__name__}({self.val})'


class FixedAlternating:
    """Returns alternating fixed values."""
    def __init__(self, val):
        self.val = np.array(val)
        self.dim = len(self.val.shape) - 1

    def __call__(self, *args, scale=1, n=None):
        if n is not None:
            return np.fromiter(
                (self(*args) for _ in range(n)),
                dtype=np.float64)
        values = self.val[tuple(args[:self.dim])]
        self.val[tuple(args[:self.dim])] = np.roll(values, 1)
        return values[-1]

    def expected(self, *args):
        """Returns the expected value for any given parameters."""
        return np.mean(self.val[tuple(args[:self.dim])])

    def distribute_time(self, n, t):
        """Returns the sum of arrival times, given n arrivals over time t."""
        return sum(i * t/(n+1) for i in range(1, n+1))

    def __repr__(self):
        return (f'{type(self).__name__}({np.array_str(self.val)})'
                .replace('\n', ''))


class Pois:
    """Returns a poisson random variable."""
    def __init__(self, mean):
        self.mean = np.array(mean)
        self.dim = len(self.mean.shape)
        self.rng = np.random.default_rng()

    def __call__(self, *args, scale=1, n=None):
        mean = self.mean[tuple(args[:self.dim])]
        try:
            return self.rng.poisson(scale * mean(*args[self.dim:], n=n))
        except TypeError:
            return self.rng.poisson(scale * mean, size=n)

    def expected(self, *args):
        """Returns the expected value for any given parameters."""
        mean = self.mean[tuple(args[:self.dim])]
        try:
            return mean(*args[self.dim:])
        except TypeError:
            return mean

    def distribute_time(self, n, t):
        """Returns the sum of arrival times, given n arrivals over time t."""
        return sum(self.rng.uniform(0, t) for _ in range(n))

    def __repr__(self):
        return f'{type(self).__name__}({np.array_str(self.mean)})'.replace('\n', '')


class TimeVarPois:
    def __init__(self, mean, time_func):
        self.mean = np.array(mean)
        self.time_func = time_func
        self.dim = len(self.mean.shape)
        self.rng = np.random.default_rng()

    def expected(self, *args):
        """Returns the expected value for any given parameters."""
        mean = self.mean[tuple(args[:self.dim])]
        time_scale = self.time_func(args[-1])
        try:
            return mean(*args[self.dim:]) * time_scale
        except TypeError:
            return mean * time_scale

    def __call__(self, *args, scale=1, n=None):
        args, t = args[:-1], args[-1]
        mean = self.mean[tuple(args[:self.dim])]
        try:
            mean = mean(*args[self.dim:])
        except TypeError:
            pass
        time_coef = np.array(self.time_func(t, scale=scale))
        size = (n, 1) if not time_coef.shape else (n,) + (time_coef).shape
        return np.sum(
            self.rng.poisson(
                time_coef * mean, size=size), axis=-1)


# class Pert:
#     """Random variable matching the PERT distribution."""
#     def __init__(self, minimum, expected, maximum):
#         self.params = (minimum, expected, maximum)
#         self.rng = PERT(minimum, expected, maximum)

#     def expected(self, *_):
#         """Returns the expected value for any given parameters."""
#         return self.params[1]

#     def __call__(self, *args, n=None):
#         if n is None:
#             return self.rng.rvs()[0]
#         return self.rng.rvs(n)


class Pert:
    """Random variable matching the PERT distribution, generated with numpy."""
    def __init__(self, minimum, expected, maximum, lamb=4):
        self.a, self.b, self.c = minimum, expected, maximum
        self.lamb = lamb
        self.alpha = 1 + (lamb * (self.b - self.a) / (self.c - self.a))
        self.beta = 1 + (lamb * (self.c - self.b) / (self.c - self.a))
        self.rng = np.random.default_rng()

    def expected(self, *_):
        """Returns the expected value for any given parameters."""
        return self.b

    def __call__(self, *args, n=None):
        return (self.rng.beta(self.alpha, self.beta, size=n)
                * (self.c - self.a) + self.a)


