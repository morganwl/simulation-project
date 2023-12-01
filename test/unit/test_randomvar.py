"""Tests random variable generators."""

from freebus.randomvar import Fixed, FixedAlternating

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
    rv = FixedAlternating([[[1,0], [2, 3]]])
    assert rv(0, 0) == 1
    assert rv(0, 0) == 0
    assert rv(0, 0, 1) == 1
    assert rv(0, 0, 2) == 0
    assert rv(0, 1) == 2
    assert rv(0, 1) == 3
    assert rv(0, 1, 1) == 2
    assert rv(0, 1, 2) == 3
