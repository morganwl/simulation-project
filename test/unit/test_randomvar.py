"""Tests random variable generators."""

import pytest
import numpy as np

from freebus.randomvar import Fixed, FixedAlternating, Pois, Pert, TimeVarPois


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
    assert np.mean(results) == pytest.approx(5, rel=0.1)
    results = rv(0, 1, n=100000)
    assert np.mean(results) == pytest.approx(9, rel=0.1)


def test_poisson_conditional_rv():
    """A random variable should be conditional upon another function."""
    rv = Pois([[Fixed(5), Fixed(9), FixedAlternating([0, 10])]])
    assert np.mean(rv(0, 0, n=10000)) == pytest.approx(5, rel=0.1)
    assert np.mean(rv(0, 1, n=10000)) == pytest.approx(9, rel=0.1)
    assert np.mean(rv(0, 2, n=10000)) == pytest.approx(5, rel=0.1)


def test_poisson_time_rv():
    """An optional extra parameter should be understood as time and
    scale the Poisson parameter."""
    rv = Pois([Fixed(5), Fixed(9)])
    assert np.mean(rv(0, scale=1, n=10000)) == pytest.approx(5, rel=0.1)
    assert (np.mean(rv(0, scale=.5, n=10000) + rv(0, scale=.5, n=10000))
            == pytest.approx(5, rel=0.1))
    assert np.mean(rv(1, scale=2, n=10000)) == pytest.approx(18, rel=0.1)
    assert (np.mean(rv(1, scale=.5, n=10000) + rv(1, scale=.5, n=10000))
            == pytest.approx(9, rel=0.1))


def test_pert_rv():
    """Tests that the PERT rv is distributed properly."""
    a, b, c = 3/60, 15/60, 120/60
    rv = Pert(a, b, c)
    vals = rv(n=100000)
    assert np.mean(vals) == pytest.approx((a + 4*b + c) / 6, rel=0.1)
    assert np.min(vals) == pytest.approx(a, rel=0.1)
    assert np.max(vals) == pytest.approx(c, rel=0.1)


def test_time_var_poisson_rv_fixed_func():
    """A time variable Poisson should have its mean scaled according to
    some universal time function."""
    # a fixed time function should behave just like a normal Poisson
    rv = TimeVarPois([2, 5], Fixed(1))
    assert np.mean(rv(0, 1, n=10000)) == pytest.approx(2, rel=0.1)
    assert np.mean(rv(0, 5, n=10000)) == pytest.approx(2, rel=0.1)
    assert np.mean(rv(1, 1, n=10000)) == pytest.approx(5, rel=0.1)
    assert np.mean(rv(1, 5, n=10000)) == pytest.approx(5, rel=0.1)


def test_time_var_poisson_rv_single_step_func():
    """A stepped time function should return two different means,
    depending on time."""
    def one_step(t, **_):
        """A simple stepped function."""
        if t < 2:
            return .25
        return .75
    rv = TimeVarPois([4, 8], one_step)
    assert np.mean(rv(0, 1, n=10000)) == pytest.approx(1, rel=0.1)
    assert np.mean(rv(0, 5, n=10000)) == pytest.approx(3, rel=0.1)
    assert np.mean(rv(1, 1, n=10000)) == pytest.approx(2, rel=0.1)
    assert np.mean(rv(1, 5, n=10000)) == pytest.approx(6, rel=0.1)


def test_time_var_poisson_rv_continuous_time_func():
    """A TimeVarPois with a continuous time function should yield a
    sum/integral over the given timescale."""
    def InterpolatedLinear(t, scale=1, step=.1):
        """Simple linear kernel function."""
        num = int(scale / step)
        return step * np.linspace(t-scale, t, endpoint=True, num=num)
    rv = TimeVarPois([4, 8], InterpolatedLinear)
    assert np.mean(rv(0, 1, scale=0, n=10000)) == pytest.approx(0, rel=0.1)
    assert np.mean(rv(0, 1, scale=1, n=10000)) == pytest.approx(2, rel=0.1)
    assert np.var(rv(0, 1, scale=1, n=10000)) == pytest.approx(2, rel=0.1)
    assert np.mean(rv(0, 4, scale=2, n=10000)) == pytest.approx(24, rel=0.1)
    assert np.var(rv(0, 4, scale=2, n=10000)) == pytest.approx(24, rel=0.1)
