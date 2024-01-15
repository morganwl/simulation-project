"""Factory functions for random variable generators."""

from warnings import warn
import inspect
from functools import lru_cache

import numpy as np
import scipy
from scipy.special import gamma


def auto_repr(cls):
    """Create __repr__ method from class's __init__ method."""

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
        """Return the value at a given set of parameters.

        Return the value at a given set of parameters, optionally
        applying a scaling factor. Return an array of n samples of the
        same variable, or a scalar value if n is None.
        """
        raise NotImplementedError

    def expected(self, *params, scale=1):
        """Return the expected value for some parameters."""
        raise NotImplementedError

    def sum_arrivals(self, n, scale, time=None):
        """Return the sum of arrival times over an interval.

        Return the sum of arrival times over an interval of length
        scale, relative to the end of that interval.
        """
        raise NotImplementedError

    def distribute_time(self, n, t):
        """
        See sum_arrivals instead.

        Given n arrivals over time t, returns the sum of arrival
        times. Deprecated.
        """
        warn('Use sum_arrivals instead.', DeprecationWarning)
        return self.sum_arrivals(n, t)


@auto_repr
class Fixed(RandomVar):
    """A fixed variable returns the same value for any number of
    inputs."""

    def __init__(self, mean):
        self.mean = np.array(mean)
        self._dim = len(self.mean.shape)

    def __call__(self, *args, scale=1, n=None):
        result = self.mean[tuple(args[:self._dim])]
        if n is not None:
            return np.ones(n) * result
        return result

    def expected(self, *params, scale=1):
        """Return the expected value for some given parameters."""
        return self.mean[tuple(params[:self._dim])]

    def sum_arrivals(self, n, scale):
        """Return the sum of arrival times, given n arrivals over time t."""
        return sum(i * t/(n+1) for i in range(1, n+1))

    def arrivals(self, alpha, beta, *params):
        """Return arrivals and sum of arrival times."""
        n = self(*params, beta, scale=beta - alpha)
        return n, self.sum_arrivals(n, beta - alpha)


@auto_repr
class FixedAlternating(RandomVar):
    """Returns alternating fixed values."""

    def __init__(self, mean):
        self.mean = np.array(mean)
        self._dim = len(self.mean.shape) - 1

    def __call__(self, *args, scale=1, n=None):
        if n is not None:
            return np.fromiter(
                (self(*args) for _ in range(n)),
                dtype=np.float64)
        values = self.mean[tuple(args[:self._dim])]
        self.mean[tuple(args[:self._dim])] = np.roll(values, 1)
        return values[-1]

    def expected(self, *params, scale=1):
        """Return the expected value for any given parameters."""
        return np.mean(self.mean[tuple(params[:self._dim])])

    def sum_arrivals(self, n, scale, time=None):
        """Return sum of arrival times, given n arrivals over an interval."""
        return sum(i * scale/(n+1) for i in range(1, n+1))

    def arrivals(self, alpha, beta, *params):
        """Return arrivals and sum of arrival times."""
        n = self(*params, beta, scale=beta - alpha)
        return n, self.sum_arrivals(n, beta - alpha)

    def __repr__(self):
        return (f'{type(self).__name__}({np.array_str(self.mean)})'
                .replace('\n', ''))


@auto_repr
class Pois(RandomVar):
    """Returns a poisson random variable."""

    def __init__(self, mean, daily_func=None):
        self.mean = np.array(mean)
        self.daily_func = daily_func
        self._daily_scale = daily_func() if daily_func else 1
        self._dim = len(self.mean.shape)
        self._rng = np.random.default_rng()

    def __call__(self, *args, scale=1, n=None):
        mean = self.mean[tuple(args[:self._dim])]
        try:
            rate = scale * mean(*args[self._dim:], n=n)
        except TypeError:
            rate = scale * mean
        return self._rng.poisson(self._daily_scale * rate, size=n)

    def expected(self, *params, scale=1):
        """Return the expected value for any given parameters."""
        mean = self.mean[tuple(params[:self._dim])]
        try:
            return mean(*params[self._dim:]) * scale
        except TypeError:
            return mean * scale

    def arrivals(self, alpha, beta, *params):
        """Return arrivals with sum of arrival times."""
        n = self(*params, beta, scale=beta - alpha)
        return n, self.sum_arrivals(n, beta - alpha, time=beta)

    def sum_arrivals(self, n, scale, time=None):
        """Returns the sum of arrival times, given n arrivals over time t."""
        return sum(self._rng.uniform(0, scale) for _ in range(n))

    def reset(self, uniform=None):
        """Generates new daily scale from daily_func, if one has been
        provided."""
        if self.daily_func:
            daily = self.daily_func(uniform=uniform)
            self._daily_scale = daily

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'{np.array_str(self.mean)})'.replace('\n', ''))


@auto_repr
class TimeVarPois(RandomVar):
    """A Poisson random variable conditioned on some function f(time)."""

    def __init__(self, mean, time_func, daily_func=None, seed=None):
        self.mean = np.array(mean)
        self.time_func = time_func
        self.daily_func = daily_func
        self.seed = seed
        self._daily_scale = daily_func() if daily_func else 1
        self._dim = len(self.mean.shape)
        self._rng = np.random.default_rng(seed=seed)
        self._arrivals = [None, None, None, None]

    def expected(self, *args, scale=1):
        """Return the expected value for any given parameters."""
        mean = self.mean[tuple(args[:self._dim])]
        time = args[-1]
        try:
            mean = mean(*args[self._dim:])
        except TypeError:
            pass
        return mean * self.time_func(time, scale=scale)

    def __call__(self, *args, scale=1, n=None):
        if scale == 0:
            return 0
        args, t = args[:-1], args[-1]
        mean = self.mean[tuple(args[:self._dim])]
        try:
            mean = mean(*args[self._dim:])
        except TypeError:
            pass

        if hasattr(self.time_func, 'components'):
            time_coefs = self.time_func.components(t-scale, t)
            if n:
                n = (n, len(time_coefs))
            components = self._rng.poisson(time_coefs
                                           * self._daily_scale
                                           * mean,
                                           size=n)
            result = np.sum(components, axis=-1)
            self._arrivals = [result, scale, t,
                              self.time_func.arrival_components(
                                  t - scale, t, components,
                                  time_coefs)]
            return result
        time_coef = self.time_func(t, scale=scale)
        return self._rng.poisson(time_coef * self._daily_scale * mean, size=n)

    def sum_arrivals(self, n, scale, time=None):
        """Returns the sum of arrival times, given n arrivals over time t."""
        if self._arrivals[:3] == [n, scale, time]:
            result = self._arrivals[-1]
            self._arrivals = [None, None, None, None]
            return result
        return self.time_func.sum_arrivals(time - scale, time, n)

    def arrivals(self, alpha, beta, *args, n=None):
        """Return arrivals with sum of relative arrival times.

        Return a discrete number of arrivals, over the time interval
        (alpha, beta], along with the sum of their arrival times,
        relative to beta.
        """
        mean = self.mean[tuple(args[:self._dim])]
        try:
            mean = mean(*args[self._dim:])
        except TypeError:
            pass
        time_coefs = self.time_func.components(alpha, beta)
        if n:
            n = (n, len(time_coefs))
        components = self._rng.poisson(time_coefs
                                       * self._daily_scale
                                       * mean,
                                       size=n)
        arrivals = np.sum(components, axis=-1)
        return arrivals, self.time_func.arrival_components(alpha, beta,
                                                           components,
                                                           time_coefs)

    def reset(self, uniform=None):
        """Generates new daily scale from daily_func, if one has been
        provided."""
        if self.daily_func:
            self._daily_scale = self.daily_func(uniform=uniform)


@auto_repr
class Pert:
    """Random variable matching the PERT distribution, generated with numpy."""

    def __init__(self, a, b, c, lamb=4, scale=None):
        self.a, self.b, self.c = a, b, c
        self.lamb = lamb
        self.scale = scale
        self._alpha = 1 + (lamb * (self.b - self.a) / (self.c - self.a))
        self._beta = 1 + (lamb * (self.c - self.b) / (self.c - self.a))
        self._rv = scipy.stats.beta(self._alpha, self._beta)
        self._rng = np.random.default_rng()

    def expected(self, *_):
        """Returns the expected value for any given parameters."""
        return self.b

    def __call__(self, scale=1, n=None):
        base = self._rv.rvs(size=n) * (self.c - self.a) + self.a
        if self.scale:
            return self.scale(base, scale)
        return base


@auto_repr
class Beta:
    """Random variable matching the beta distribution."""

    def __init__(self, a, b, bias=0):
        self.a = a
        self.b = b
        self.bias = bias
        self._rng = np.random.default_rng()

    def expected(self, *_):
        return self.a / (self.a + self.b) + self.bias

    def transform(self, u):
        """Use a uniform random variable to generate a beta random
        variable."""
        return scipy.stats.beta.ppf(u, self.a, self.b) + self.bias

    def __call__(self, n=None, uniform=None):
        if uniform:
            return self.transform(uniform)
        return self._rng.beta(self.a, self.b, size=n) + self.bias


@auto_repr
class Gamma(RandomVar):
    """Random variable matching the gamma distribution."""

    def __init__(self, k, theta, seed=None):
        self.k, self.theta = k, theta
        rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self._rng = rng

    def expected(self, *_):
        return self.k * self.theta

    def __call__(self, n=None):
        return self._rng.gamma(self.k, self.theta, size=n)


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


@auto_repr
class SumOfFunctionKernel:
    """A kernel that returns the sum of a list of arbitrary functions
    over time t."""

    def __init__(self, funcs: list):
        self.funcs = funcs

    def __call__(self, t, scale=1, step=0.1):
        if scale == 0:
            return np.array([])
        num = int(np.ceil(scale / step))
        step = scale / num
        segments = np.linspace(t - scale, t, endpoint=True, num=num)
        return step * sum(f(segments) for f in self.funcs)


@auto_repr
class SumOfDistributionKernel:
    """A kernel that returns the sum of of a list of cumulative
    distribution functions over time t."""

    def __init__(self, funcs: list, seed=None):
        self.funcs = funcs
        self.seed = seed
        self._area = np.array([f.area for f in funcs])
        self._arrival_cache = np.zeros(4)
        self._rng = np.random.default_rng(seed=seed)

    @lru_cache
    def __call__(self, t, scale=1):
        return np.sum([f(t) - f(t - scale) for f in self.funcs])

    def components(self, alpha, beta):
        """Return the raw component values over a given interval."""
        return np.fromiter((f(beta) - f(alpha) for f in self.funcs),
                           dtype=np.float64, count=len(self.funcs))

    def arrival_components(self, alpha, beta, components, coefs):
        """Return sum of arrival times from multiple poisson components."""
        return sum(n * beta - np.sum(f.inverse(
            p / f.area * self._rng.uniform(size=n) + f.cdf(alpha)))
            for n, f, p in zip(components, self.funcs, coefs))
        # for n, f, p in zip(components, self.funcs, coefs):
        #     u = self._rng.uniform(size=n)

    @lru_cache
    def _cdf(self, t):
        return sum(f(t) for f in self.funcs)

    def sum_arrivals(self, alpha, beta, n):
        if (self._arrival_cache[:3] == [alpha, beta, n]).all():
            return self._arrival_cache[3]
        p_alpha = self._cdf(alpha)
        p = self._cdf(beta) - p_alpha

        def distance(t, u):
            return (self._cdf(t) - p_alpha) / p - u
        return sum(beta -
                   self.find_zero(distance, (alpha, beta),
                                  np.random.uniform(),)
                   for _ in range(n))

    @staticmethod
    def find_zero(func, bounds, u):
        """Find the zero of a monotonic function."""
        lx, ux = bounds
        assert 0 <= u < 1
        mx = lx + (ux - lx) * u
        my = func(mx, u)
        while ux - lx > .25:
            if my > 0:
                tmp = mx
                mx = ux - (ux - lx) / 2
                ux = tmp
                my = func(mx, u)
            else:
                tmp = mx
                mx = lx + (ux - lx) / 2
                lx = tmp
                my = func(mx, u)
        assert bounds[0] <= mx <= bounds[1]
        return mx


@auto_repr
class SumOf:
    """A callable sum of callable objects."""

    def __init__(self, funcs: list):
        self.funcs = funcs

    def __call__(self, *args, **kwargs):
        return sum(f(*args, **kwargs) for f in self.funcs)


# pylint: disable=too-few-public-methods
@auto_repr
class GammaTimeFunc:
    """A time function based on a gamma distribution."""

    def __init__(self, k, theta, area=1):
        self.k = k
        self.theta = theta
        self.area = area
        self._c = area / (gamma(k) * theta ** k) / 60

    def __call__(self, x):
        return scipy.stats.gamma.cdf(x, self.k, self.theta) * self.area

    def inverse(self, u):
        return scipy.stats.gamma.ppf(u, self.k, self.theta)


@auto_repr
class BetaTimeFunc:
    """A time function based on a beta distribution."""

    def __init__(self, a, b, area=1, pdf=False):
        self.a = a
        self.b = b
        self.area = area
        self.pdf = pdf
        self._rv = scipy.stats.beta(a, b)

    def scale_input(self, t):
        """Scales an input in minutes to a fraction of a day."""
        return t / 1440

    def scale_output(self, p):
        """Scales a fraction of a day in [0,1] to a time in minutes."""
        return p * 1440

    def __call__(self, t):
        if self.pdf:
            return self._pdf(t)
        return self._rv.cdf(self.scale_input(t)) * self.area

    def _pdf(self, t):
        """Return the probability density function at time t."""
        return self._rv.pdf(self.scale_input(t)) * self.area

    def inverse(self, u):
        """Return the corresponding time t for the probability u."""
        return self.scale_output(self._rv.ppf(u))

    def cdf(self, t):
        """Return the unscaled cumulative probability for time t."""
        return self._rv.cdf(self.scale_input(t))
