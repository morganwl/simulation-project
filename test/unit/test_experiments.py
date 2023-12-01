"""Test the Experiment object class."""

from freebus.experiments import Experiment
from freebus.randomvar import Fixed, FixedAlternating, Pois

def test_experiment_repr_simple():
    """Tests that an Experiment object has a stable, useful,
    representation."""
    args1 = [
            [2], [1, 0],
            Fixed(1),
            FixedAlternating([[[1,0], [0,0]]]),
            Fixed([[0, 1]]),
            Fixed(1),
            Fixed(1),
            [[0, 10]],
            ['loading-time', 'moving-time', 'holding-time', 'total-passengers'],
            ]
    args2 = args1[:]
    args2[-2] = [[0, 10, 20]]
    exp1 = Experiment(*args1)
    exp2 = Experiment(*args1)
    exp3 = Experiment(*args2)
    assert exp1 != exp2
    assert str(exp1) == str(exp2)
    assert str(exp1) != str(exp3)

def test_experiment_repr_poisson():
    """Tests that an Experiment object has a stable representation when
    containing Poisson parameters."""
    args1 = [
            [2], [1, 0],
            Fixed(1),
            Pois([[1, 0]]),
            Pois([[Fixed(0), Fixed(1)]]),
            Fixed(1),
            Fixed(1),
            [[0, 10]],
            ['loading-time', 'moving-time', 'holding-time', 'total-passengers'],
            ]
    args2 = args1[:]
    args2[-2] = [[0, 10, 20]]
    exp1 = Experiment(*args1)
    exp2 = Experiment(*args1)
    exp3 = Experiment(*args2)
    assert exp1 != exp2
    assert str(exp1) == str(exp2)
    assert str(exp1) != str(exp3)

def test_experiment_checksum():
    """Tests that an experiment yields a stable checksum."""
    args1 = [
            [2], [1, 0],
            Fixed(1),
            FixedAlternating([[[1,0], [0,0]]]),
            Fixed([[0, 1]]),
            Fixed(1),
            Fixed(1),
            [[0, 10]],
            ['loading-time', 'moving-time', 'holding-time', 'total-passengers'],
            ]
    args2 = args1[:]
    args2[-2] = [[0, 10, 20]]
    exp1 = Experiment(*args1)
    exp2 = Experiment(*args1)
    exp3 = Experiment(*args2)
    assert exp1 != exp2
    assert exp1.checksum() == exp2.checksum()
    assert exp1.checksum() != exp3.checksum()
