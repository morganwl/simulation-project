"""Tests random variable generators."""

from pathlib import Path

import pytest
from pytest import approx
import numpy as np

from freebus.randomvar import Fixed, FixedAlternating, Pois, Pert, \
    TimeVarPois, IndicatorKernel, GammaTimeFunc, \
    BetaTimeFunc, SumOfDistributionKernel, Beta


def test_fixed_rv():
    """A fixed rv should return a single value for all inputs."""
    rv = Fixed(2)
    assert rv(0, 1, 2) == 2
    assert rv(1) == 2
    assert rv(4) == 2
    rv = Fixed(3)
    assert rv(0, 1, 2) == 3
    assert rv(1) == 3
    assert rv(4) == 3
    assert str(rv) == 'Fixed(3)'


def test_fixed_alternating_matrix_rv():
    """A FixedAlternating rv should return an alternating value, defined
    for each cell in an (r,s) matrix."""
    rv = FixedAlternating([[[1, 0], [2, 3]]])
    assert rv(0, 0) == 1
    assert rv(0, 0) == 0
    assert rv(0, 0, 1) == 1
    assert rv(0, 0, 2) == 0
    assert rv(0, 1) == 2
    assert rv(0, 1) == 3
    assert rv(0, 1, 1) == 2
    assert rv(0, 1, 2) == 3

def test_fixed_alternating_scalar_rv():
    """A FixedAlternating rv should also accept a scalar parameter."""
    rv = FixedAlternating([0,1])
    assert rv() == 0
    assert rv() == 1
    assert rv() == 0

def test_fixed_alternating_array():
    """A FixedAlternating rv should be able to return an array of n
    elements."""
    rv = FixedAlternating([0,1])
    assert (rv(n=2) == [0, 1]).all()


def test_poisson_rv():
    """A poisson rv should behave according to a Poisson process."""
    rv = Pois([[5, 9]])
    results = rv(0, 0, n=100000)
    assert np.mean(results) == approx(5, rel=0.1)
    results = rv(0, 1, n=100000)
    assert np.mean(results) == approx(9, rel=0.1)


def test_poisson_conditional_rv():
    """A random variable should be conditional upon another function."""
    rv = Pois([[Fixed(5), Fixed(9), FixedAlternating([0, 10])]])
    assert np.mean(rv(0, 0, n=10000)) == approx(5, rel=0.1)
    assert np.mean(rv(0, 1, n=10000)) == approx(9, rel=0.1)
    assert np.mean(rv(0, 2, n=10000)) == approx(5, rel=0.1)


def test_poisson_time_rv():
    """An optional extra parameter should be understood as time and
    scale the Poisson parameter."""
    rv = Pois([Fixed(5), Fixed(9)])
    assert np.mean(rv(0, scale=1, n=10000)) == approx(5, rel=0.1)
    assert (np.mean(rv(0, scale=.5, n=10000) + rv(0, scale=.5, n=10000))
            == approx(5, rel=0.1))
    assert np.mean(rv(1, scale=2, n=10000)) == approx(18, rel=0.1)
    assert (np.mean(rv(1, scale=.5, n=10000) + rv(1, scale=.5, n=10000))
            == approx(9, rel=0.1))


def test_pert_rv():
    """Tests that the PERT rv is distributed properly."""
    a, b, c = 3/60, 15/60, 120/60
    rv = Pert(a, b, c)
    vals = rv(n=100000)
    assert np.mean(vals) == approx((a + 4*b + c) / 6, rel=0.1)
    assert np.min(vals) == approx(a, rel=0.1)
    assert np.max(vals) == approx(c, rel=0.1)


def test_scaled_pert_rv():
    """Tests that a Pert variable can be created with a scaling function."""
    a, b, c = 3/60, 15/60, 120/60

    def scaling_func(x, y):
        return x * y**2
    rv = Pert(a, b, c, scale=scaling_func)
    unscaled_vals = rv(n=100000)
    scaled_vals = rv(scale=4, n=100000)
    assert np.mean(unscaled_vals) == approx((a + 4*b + c) / 6, rel=0.1)
    assert np.mean(scaled_vals) == approx(
        scaling_func((a + 4*b + c) / 6, 4), rel=0.1)


def test_time_var_poisson_rv_fixed_func():
    """A time variable Poisson should have its mean scaled according to
    some universal time function."""
    # a fixed time function should behave just like a normal Poisson
    rv = TimeVarPois([2, 5], Fixed(1))
    assert np.mean(rv(0, 1, n=10000)) == approx(2, rel=0.1)
    assert np.mean(rv(0, 5, n=10000)) == approx(2, rel=0.1)
    assert np.mean(rv(1, 1, n=10000)) == approx(5, rel=0.1)
    assert np.mean(rv(1, 5, n=10000)) == approx(5, rel=0.1)


def test_gamma_time_func_integral():
    """Tests that a GammaTimeFunc integrates to 1."""
    shape = 9
    scale = 7/shape
    func = GammaTimeFunc(shape, scale)
    kernel = SumOfDistributionKernel([func])
    expected = 1
    result = np.sum(kernel(24*60, 24*60))
    assert result == approx(expected, rel=0.01)


def test_gamma_time_func_stable():
    """Tests that GammaTimeFunc does not mutate an input array."""
    x = np.array([1, 2])
    func = GammaTimeFunc(9, 7/9)
    r1 = func(x)
    r2 = func(x)
    assert (r1 == r2).all()


def test_gamma_time_func_sums_integral():
    """Tests that GammaTimeFuncs can be scaled and still integrate to
    1."""
    shape1 = 9
    scale1 = 7/shape1
    func1 = GammaTimeFunc(shape1, scale1, area=.7)

    shape2 = 9
    scale2 = 7/shape2
    func2 = GammaTimeFunc(shape2, scale2, area=.3)

    kernel = SumOfDistributionKernel([func1, func2])
    expected = 1
    result = np.sum(kernel(24*60, 24*60))
    assert result == approx(expected, rel=0.01)


def test_beta_time_func_integral():
    """Tests that a BetaTimeFunc integrates to 1."""
    a = 5
    b = a - (13/24*a)
    func = BetaTimeFunc(a, b)
    kernel = SumOfDistributionKernel([func])
    expected = 1
    result = np.sum(kernel(24*60, 24*60))
    assert result == approx(expected, rel=0.01)


def test_beta_time_func_sums_integral():
    """Tests that BetaTimeFuncs can be scaled and still integrate to
    1."""
    shape1 = 9
    scale1 = 5
    func1 = BetaTimeFunc(shape1, scale1, area=.7)

    shape2 = 9
    scale2 = 5
    func2 = BetaTimeFunc(shape2, scale2, area=.3)

    kernel = SumOfDistributionKernel([func1, func2])
    expected = 1
    result = np.sum(kernel(24*60, 24*60))
    assert result == approx(expected, rel=0.0001)


@pytest.mark.parametrize(['RV', 'params', 'mean'],
                           [[Pois, [12], 12],
                            [TimeVarPois, [12, lambda x, scale: .5], 6]])
@pytest.mark.parametrize('time', [30, 60, 120, 240, 480, 1200])
def test_rv_daily_func(RV, params, mean, time, ReturnFrom):
    daily_values = [.5, .75, 1.25, 1.5]
    daily_func = ReturnFrom(daily_values)
    rv = RV(*params, daily_func=daily_func)
    results = []
    for _ in daily_values:
        results.append(np.mean([rv(time) for _ in range(500)]))
        rv.reset()
    assert results == approx([dv * mean for dv in daily_values],
                             rel=0.1)


def rejection_estimate_poisson_arrivals(alpha, beta, n, rv, num=50):
    """Generate a random variable equal to the sum of arrival times
    (relative to beta), given n arrivals within the interval (alpha,
    beta)."""
    interval = (beta - alpha) / num
    times = np.linspace(alpha, beta, num=num)
    k = 0
    while np.sum(k) != n:
        k = np.fromiter((rv(ti, scale=interval)
                         for ti in times),
                        dtype=np.int32,
                        count=num)
    return np.sum((beta - times) * k)


@pytest.mark.skip(reason='very slow, not essential.')
@pytest.mark.parametrize(['lam', 'scale', 'n'],
                         [(1, 1, 0),
                          (1, 1, 1),
                          (1, 5, 4),
                          (3, 5, 13),])
@pytest.mark.parametrize('time', np.linspace(30, 24*60, num=3))
def test_rejection_estimate_poisson_arrivals(time, lam, scale, n):
    """Test that rejection estimate of poisson arrival times works with
    a homogeneous poisson process."""
    rv = Pois(lam)
    result = np.mean([
        rejection_estimate_poisson_arrivals(time - scale, time, n, rv)
        for _ in range(250)])
    expected = np.mean([rv.sum_arrivals(n, scale) for _ in range(1000)])
    assert result == approx(expected, rel=0.1)


@pytest.mark.parametrize(['time', 'lam', 'scale', 'n'],
                         [(120, 540, 90, 2),
                          (420, 220, 30, 10),
                          (300, 75, 30, 2),
                          (800, 215, 15, 2)])
@pytest.mark.parametrize('func',
                         [SumOfDistributionKernel([
                             BetaTimeFunc(6, 14, area=0.5),
                             BetaTimeFunc(4, 2, area=0.5),])],
                         )
def test_time_var_poisson_arrival_times(time, lam, scale, n, func):
    """Test that a time variable poisson process accurately distributes
    arrival times across an interval."""
    rv = TimeVarPois(lam, func, seed=1)
    result = np.mean([
        rv.sum_arrivals(n, scale, time=time)
        for _ in range(2000)])
    expected = np.mean([
        rejection_estimate_poisson_arrivals(time - scale, time, n, rv)
        for _ in range(100)])
    assert result == approx(expected, rel=0.1)


@pytest.mark.parametrize(['time', 'lam', 'scale', 'n'],
                         [(120, 540, 90, 2),
                          (420, 220, 30, 10),
                          (300, 75, 30, 2),
                          (800, 215, 15, 2)])
@pytest.mark.parametrize('func',
                         [SumOfDistributionKernel([
                             BetaTimeFunc(6, 14, area=0.5),
                             BetaTimeFunc(4, 2, area=0.5),])],
                         )
def test_analytic_poisson_arrival_times(time, lam, scale, n, func):
    """Test that the analytic arrival times produce comparable results
    to the gradient descent times."""
    rv = TimeVarPois(lam, func, seed=1)
    result = []
    while len(result) < 1000:
        if rv(time, scale=scale) != n:
            continue
        result.append(rv.sum_arrivals(n, scale, time=time))
    result = np.mean(result)
    expected = np.mean([
        rv.sum_arrivals(n, scale, time=time)
        for _ in range(1000)])
    assert result == approx(expected, rel=0.05)


@pytest.mark.parametrize(['a', 'b'],
                         [(2, 2),
                          (9, 10),
                          (5, .2)])
@pytest.mark.parametrize('bias',
                         [0, .25, .5, .75, 1])
def test_beta_antithetic_variables(a, b, bias):
    """Beta should have same mean but lower variance with antithetic
    uniform variables."""
    rng = np.random.default_rng()

    def antithetic_rv(rv):
        u = rng.uniform()
        return np.mean([rv.transform(u), rv.transform(1 - u)])
    rv = Beta(a, b, bias=bias)
    result = [antithetic_rv(rv) for _ in range(500)]
    mean, std = np.mean(result), np.std(result)
    expected = [rv() for _ in range(1000)]
    exp_mean, exp_std = np.mean(expected), np.std(expected)
    assert exp_mean == approx(a / (a + b) + bias, rel=0.05)
    assert mean == approx(exp_mean, rel=0.05)
    assert std < exp_std
