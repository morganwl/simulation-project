"""Tests random variable generators."""

import pytest
from pytest import approx
import numpy as np
import scipy

from freebus.randomvar import Fixed, FixedAlternating, Pois, Pert, \
    TimeVarPois, IndicatorKernel, SumOfFunctionKernel, GammaTimeFunc, \
    BetaTimeFunc, SumOfDistributionKernel


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


def test_time_var_poisson_rv_fixed_func():
    """A time variable Poisson should have its mean scaled according to
    some universal time function."""
    # a fixed time function should behave just like a normal Poisson
    rv = TimeVarPois([2, 5], Fixed(1))
    assert np.mean(rv(0, 1, n=10000)) == approx(2, rel=0.1)
    assert np.mean(rv(0, 5, n=10000)) == approx(2, rel=0.1)
    assert np.mean(rv(1, 1, n=10000)) == approx(5, rel=0.1)
    assert np.mean(rv(1, 5, n=10000)) == approx(5, rel=0.1)


def test_time_var_poisson_rv_single_step_func():
    """A stepped time function should return two different means,
    depending on time."""
    def one_step(t, **_):
        """A simple stepped function."""
        if t < 2:
            return .25
        return .75
    rv = TimeVarPois([4, 8], one_step)
    assert np.mean(rv(0, 1, n=10000)) == approx(1, rel=0.1)
    assert np.mean(rv(0, 5, n=10000)) == approx(3, rel=0.1)
    assert np.mean(rv(1, 1, n=10000)) == approx(2, rel=0.1)
    assert np.mean(rv(1, 5, n=10000)) == approx(6, rel=0.1)


def test_time_var_poisson_rv_continuous_time_func():
    """A TimeVarPois with a continuous time function should yield a
    sum/integral over the given timescale."""
    def interpolated_linear(t, scale=1, step=.1):
        """Simple linear kernel function."""
        num = int(scale / step)
        return step * np.linspace(t-scale, t, endpoint=True, num=num)
    rv = TimeVarPois([4, 8], interpolated_linear)
    assert np.mean(rv(0, 1, scale=0, n=10000)) == approx(0, rel=0.1)
    assert np.mean(rv(0, 1, scale=1, n=10000)) == approx(2, rel=0.1)
    assert np.var(rv(0, 1, scale=1, n=10000)) == approx(2, rel=0.1)
    assert np.mean(rv(0, 4, scale=2, n=10000)) == approx(24, rel=0.1)
    assert np.var(rv(0, 4, scale=2, n=10000)) == approx(24, rel=0.1)


def test_indicator_kernel():
    """An indicator kernel should return some fixed non-zero number
    between two bounds, and zero outside of those bounds, such that the
    sum across the entire space is equal to a given volume."""
    kernel = IndicatorKernel(volume=1, lower=1, upper=2)
    assert np.sum(kernel(1, scale=1)) == approx(0)
    assert np.sum(kernel(1.5, scale=1)) == approx(.5)
    assert np.sum(kernel(2, scale=1)) == approx(1)
    assert np.sum(kernel(3, scale=3)) == approx(1)
    assert np.sum(kernel(2.75, scale=1, step=.05)) == approx(.25)


def test_sum_of_function_kernel_with_additive_inverses():
    """A kernel with a function and its negative inverse should have
    zero area."""
    kernel = SumOfFunctionKernel([lambda x: x, lambda x: -x])
    result = np.sum(kernel(10, 5))
    assert result == 0


def test_sum_of_function_kernel_with_linear_funcs():
    """Tests that a kernel with two linear functions has an area
    consistent with the sum of their definite integrals."""
    kernel = SumOfFunctionKernel([lambda x: x, lambda x: 2*x])
    t = 4
    scale = 3
    result = np.sum(kernel(t, scale))
    definite_integral = (t**2 - (t - scale)**2)/2 + t**2 - (t - scale)**2
    assert result == approx(definite_integral)


def test_sum_of_function_kernel_with_nonlinear_funcs():
    """Tests that a kernel with three non-linear functions has an area
    consistent with the sum of their definite integrals."""
    kernel = SumOfFunctionKernel([lambda x: 3*x**2,
                                  lambda x: x**3,
                                  lambda x: 5*x**4])
    t = 6
    scale = 2
    step = 0.01
    # the polynomial function requires more segments
    result = np.sum(kernel(t, scale, step=step))
    definite_integral = sum(
        [t**3 - (t - scale)**3,
         (t**4 - (t - scale)**4) / 4,
         t**5 - (t - scale)**5])
    assert result == approx(definite_integral, rel=0.001)


def test_time_var_poisson_with_sum_of_linear_funcs():
    """Tests that a TimeVarPois with a SumOfFunctionKernel returns the
    expected mean value."""
    rv = TimeVarPois(5, SumOfFunctionKernel(
        [lambda x: x, lambda x: 2*x]))
    t = 5
    scale = 3
    expected = 5 * sum([
        (t**2 - (t - scale)**2) / 2,
        (t**2 - (t - scale)**2)])
    result = np.mean(rv(t, scale=scale, n=10000))
    assert result == approx(expected, rel=0.01)


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


def test_beta_time_func_stable():
    """Tests that BetaTimeFunc does not mutate an input array."""
    x = np.array([1, 2])
    func = BetaTimeFunc(5, 5)
    r1 = func(x)
    r2 = func(x)
    assert (r1 == r2).all()


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
