"""Factory functions for random variable generators."""

from warnings import warn
import inspect

import numpy as np


def auto_repr(cls):
    """Create a __repr__ method from the signature of the __init__
    method of a class."""
    def __repr__(self):
        params = []
        for k, p in inspect.signature(cls.__init__).parameters.items():
            if k == 'self':
                continue
            if p.default is not p.empty:
                if p.default == getattr(self, k):
                    continue
                params.append(f'{k}={getattr(self, k)}')
            else:
                params.append(f'{getattr(self, k)}')
        return ''.join([
            cls.__name__, '(',
            ', '.join(params), ')'])
    cls.__repr__ = __repr__
    return cls


class RandomVar:
    """A callable object initialized with specific values."""
    def __call__(self, *args, scale=1, n=None):
        """Returns the value at a given set of parameters, optionally
        scaled. Returns an array of n samples of the same variable, or a
        scalar value if n is None."""
        raise NotImplementedError

    def expected(self, *args):
        """Returns the expected (mean) value for any given
        parameters."""
        raise NotImplementedError

    def sum_arrivals(self, n, t):
        """Given n arrivals over time t, returns the sum of arrival
        times."""
        raise NotImplementedError

    def distribute_time(self, n, t):
        """
        Deprecated. See sum_arrivals instead.

        Given n arrivals over time t, returns the sum of arrival
        times.
        """
        warn('Use sum_arrivals instead.', DeprecationWarning)
        return self.sum_arrivals(n, t)


@auto_repr
class Fixed(RandomVar):
    """A fixed variable returns the same value for any number of
    inputs."""
    def __init__(self, val):
        self.val = np.array(val)
        self._dim = len(self.val.shape)

    def __call__(self, *args, scale=1, n=None):
        result = self.val[tuple(args[:self._dim])]
        if n is not None:
            return np.ones(n) * result
        return result

    def expected(self, *args):
        """Returns the expected value for any given parameters."""
        return self.val[tuple(args[:self._dim])]

    def sum_arrivals(self, n, t):
        """Returns the sum of arrival times, given n arrivals over time t."""
        return sum(i * t/(n+1) for i in range(1, n+1))


@auto_repr
class FixedAlternating(RandomVar):
    """Returns alternating fixed values."""
    def __init__(self, val):
        self.val = np.array(val)
        self._dim = len(self.val.shape) - 1

    def __call__(self, *args, scale=1, n=None):
        if n is not None:
            return np.fromiter(
                (self(*args) for _ in range(n)),
                dtype=np.float64)
        values = self.val[tuple(args[:self._dim])]
        self.val[tuple(args[:self._dim])] = np.roll(values, 1)
        return values[-1]

    def expected(self, *args):
        """Returns the expected value for any given parameters."""
        return np.mean(self.val[tuple(args[:self._dim])])

    def sum_arrivals(self, n, t):
        """Returns the sum of arrival times, given n arrivals over time t."""
        return sum(i * t/(n+1) for i in range(1, n+1))

    def __repr__(self):
        return (f'{type(self).__name__}({np.array_str(self.val)})'
                .replace('\n', ''))


@auto_repr
class Pois(RandomVar):
    """Returns a poisson random variable."""
    def __init__(self, mean):
        self.mean = np.array(mean)
        self._dim = len(self.mean.shape)
        self._rng = np.random.default_rng()

    def __call__(self, *args, scale=1, n=None):
        mean = self.mean[tuple(args[:self._dim])]
        try:
            return self._rng.poisson(scale * mean(*args[self._dim:], n=n))
        except TypeError:
            return self._rng.poisson(scale * mean, size=n)

    def expected(self, *args):
        """Returns the expected value for any given parameters."""
        mean = self.mean[tuple(args[:self._dim])]
        try:
            return mean(*args[self._dim:])
        except TypeError:
            return mean

    def sum_arrivals(self, n, t):
        """Returns the sum of arrival times, given n arrivals over time t."""
        return sum(self._rng.uniform(0, t) for _ in range(n))

    def __repr__(self):
        return f'{type(self).__name__}({np.array_str(self.mean)})'.replace('\n', '')


@auto_repr
class TimeVarPois(RandomVar):
    def __init__(self, mean, time_func):
        self.mean = np.array(mean)
        self.time_func = time_func
        self._dim = len(self.mean.shape)
        self._rng = np.random.default_rng()

    def expected(self, *args):
        """Returns the expected value for any given parameters."""
        mean = self.mean[tuple(args[:self._dim])]
        time_scale = self.time_func(args[-1])
        try:
            return mean(*args[self._dim:]) * time_scale
        except TypeError:
            return mean * time_scale

    def __call__(self, *args, scale=1, n=None):
        args, t = args[:-1], args[-1]
        mean = self.mean[tuple(args[:self._dim])]
        try:
            mean = mean(*args[self._dim:])
        except TypeError:
            pass
        time_coef = np.array(self.time_func(t, scale=scale))
        if n is None:
            size = time_coef.shape
        else:
            size = (n, 1) if not time_coef.shape else (n,) + (time_coef).shape
        return np.sum(
            self._rng.poisson(
                time_coef * mean, size=size), axis=-1)

    def sum_arrivals(self, n, t):
        """Returns the sum of arrival times, given n arrivals over time t."""
        return sum(self._rng.uniform(0, t) for _ in range(n))


@auto_repr
class Pert(RandomVar):
    """Random variable matching the PERT distribution, generated with numpy."""
    def __init__(self, a, b, c, lamb=4):
        self.a, self.b, self.c = a, b, c
        self.lamb = lamb
        self._alpha = 1 + (lamb * (self.b - self.a) / (self.c - self.a))
        self._beta = 1 + (lamb * (self.c - self.b) / (self.c - self.a))
        self._rng = np.random.default_rng()

    def expected(self, *_):
        """Returns the expected value for any given parameters."""
        return self.b

    def __call__(self, *args, n=None):
        return (self._rng.beta(self._alpha, self._beta, size=n)
                * (self.c - self.a) + self.a)


@auto_repr
class IndicatorKernel:
    """An indicator kernel has a value of either x or 0, where the
    integral over the bounds is equal to volume."""
    def __init__(self, volume, lower, upper):
        self.value = volume / (upper - lower)
        self.volume = volume
        self.lower = lower
        self.upper = upper

    def __call__(self, t, scale=1, step=.01):
        if scale == 0:
            return np.array([])
        num = int(np.ceil(scale / step))
        step = scale / num
        segments = np.linspace(t - scale, t, endpoint=False, num=num)
        return step * ((self.lower <= segments) & (segments < self.upper))

    def __repr__(self):
        return ''.join([f'{type(self).__name__}(',
                        ', '.join(f'{key}={getattr(self, key)}'
                                  for key in ['volume', 'lower', 'upper']),
                        ')'])
