"""Experiment parameters."""

from dataclasses import dataclass, field
from zlib import crc32

import numpy as np
from scipy.special import beta, gamma
import scipy.stats

from .randomvar import RandomVar, Fixed, FixedAlternating, Pois, Pert, \
    TimeVarPois, SumOfDistributionKernel, GammaTimeFunc, BetaTimeFunc, \
    IndicatorKernel, auto_repr, Gamma, SumOf, Beta


class Headers:
    # pylint: disable=too-few-public-methods
    # this class is merely a collection of constant tuples
    """Preset collections of experimental values to record for a
    trial."""
    SIMPLE = ('waiting-time', 'loading-time', 'moving-time',
              'holding-time', 'total-passengers')
    EXTENDED = SIMPLE + (('last-event', 'median-load', 'extreme-load',)
                         + tuple(f'passengers-{i}' for i in range(24)))


@dataclass
class Transfer:
    """Parameters for a transfer between two routes."""
    fr_route: int
    fr_stop: int
    to_route: int
    to_stop: int
    rate: float
    waiting: int = 0
    last_time: float = 0


@dataclass
class Routes:
    """Parameters for external conditions for a predefined collection of
    bus-stops."""
    routes: list[int]
    distance: list[list[int]]
    traffic: list[list[int]]
    demand_loading: RandomVar
    demand_unloading: RandomVar
    transfers: list[Transfer] = field(default_factory=list)

    def __post_init__(self):
        self.transfers = [Transfer(*t) for t in self.transfers]

    def reset(self):
        """Resets any per-trial parameters."""
        try:
            self.traffic.reset()
        except AttributeError:
            pass


@dataclass
class WorldModeling:
    """Parameters for modeling various events."""
    time_loading: RandomVar
    time_unloading: RandomVar


@dataclass
class Schedule:
    """Parameters for bus schedule policies."""
    schedule: list[list[float]]


@dataclass
class Node:
    """A binary traffic node."""
    time: float
    val: float
    left: 'Node' = None
    right: 'Node' = None


class TrafficModel:
    """Models traffic volume with consistency."""
    def __init__(self, rand_scale, time_func=None,
                 daily_func=None):
        self.rand_scale = rand_scale
        if time_func is None:
            time_func = lambda x: 0
        self.time_func = time_func
        if daily_func is None:
            daily_func = lambda: 1
        self.daily_func = daily_func
        self._daily_scale = daily_func()
        self.time_trees = {}

    def __call__(self, route, stop, t):
        earlier, later = self.find_neighbors(route, stop, t)
        weight = 0
        val = 0
        if earlier:
            w = .5 * .9**(t - earlier.time)
            weight += w
            val += w * earlier.val
        if later:
            w = .5 * .9**(later.time - t)
            weight += w
            val += w * later.val
        val += ((1 - weight) *
                (1 + self.time_func(t))
                 ** (self.rand_scale() * self._daily_scale))
        self.insert(route, stop, t, val)
        return val

    def find_neighbors(self, route, stop, t):
        """Find the two closest nodes to time t. If t is already in the
        tree, these are both t."""
        earlier = None
        later = None
        node = self.time_trees.get((route, stop))
        while node is not None:
            if node.time == t:
                return node, node
            if node.time < t:
                earlier = node
                node = node.right
            else:
                later = node
                node = node.left
        return earlier, later

    def insert(self, route, stop, t, val):
        """Inserts a node in the time tree."""
        node = self.time_trees.get((route, stop))
        while node is not None:
            if node.time == t:
                return
            if node.time > t:
                if node.left is None:
                    node.left = Node(t, val)
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = Node(t, val)
                    return
                node = node.right
        self.time_trees[(route, stop)] = Node(t, val)

    def reset(self):
        """Reset per-trial values in the traffic model."""
        self.time_trees = {}
        self._daily_scale = self.daily_func()

    def fix(self, route, stop, t, val):
        self.insert(route, stop, t, val)

    def __repr__(self):
        return f'{type(self).__name__}()'


class Experiment:
    """Object containing experimental parameters."""
    def __init__(self,
                 routes,
                 time_loading, time_unloading,
                 schedule,
                 headers,
                 speed=20/60):
        self._routes = routes
        self.time_loading = time_loading
        self.time_unloading = time_unloading
        self.schedule = schedule
        self.headers = headers
        self.speed = speed

    def checksum(self):
        """Returns a crc32 checksum of the experimental parameters as a
        hexadecimal string."""
        return hex(crc32(str(self).encode('utf8')))[2:]

    @property
    def routes(self):
        return self._routes.routes

    @property
    def distance(self):
        return self._routes.distance

    @property
    def traffic(self):
        return self._routes.traffic

    @property
    def demand_loading(self):
        return self._routes.demand_loading

    @property
    def demand_unloading(self):
        return self._routes.demand_unloading

    def get_transfers(self, route, stop):
        """Returns a list of all transfers from route, stop."""
        return [t for t in self._routes.transfers
                if t.fr_route == route and t.fr_stop == stop]

    def get_transfers_to(self, route, stop):
        """Returns a list of all transfers to route, stop."""
        return [t for t in self._routes.transfers
                if t.to_route == route and t.to_stop == stop]

    def reset(self):
        """Resets any per-trial parameters."""
        self._routes.reset()

    def __repr__(self):
        return (
            f'{type(self).__name__}('
            f'{self.routes}, {self.distance}, '
            f'{self.schedule}, {self.headers}, '
            f'traffic={self.traffic}, demand_loading={self.demand_loading}, '
            f'demand_unloading={self.demand_unloading}, '
            f'time_loading={self.time_loading})')


def b35_schedule():
    """Return the b35 schedule."""
    return [[30, 80, 130, 180, 225, 265, 295, 318, 339, 355, 375, 382,
             389, 396, 403, 411, 418, 426, 433, 441, 448, 456, 463, 471,
             478, 486, 493, 501, 509, 517, 526, 538, 550, 562, 574, 586,
             598, 610, 622, 634, 646, 658, 670, 682, 694, 706, 718, 730,
             742, 754, 764, 774, 784, 794, 804, 814, 824, 834, 844, 854,
             864, 874, 881, 889, 896, 904, 911, 919, 926, 934, 942, 950,
             958, 966, 974, 982, 990, 998, 1006, 1014, 1022, 1029, 1037,
             1044, 1051, 1058, 1065, 1072, 1079, 1086, 1094, 1103, 1112,
             1121, 1130, 1139, 1148, 1157, 1166, 1175, 1185, 1197, 1209,
             1221, 1233, 1245, 1257, 1269, 1281, 1296, 1315, 1331, 1343,
             1355, 1367, 1379, 1392, 1405, 1420, 1436]]


def get_builtin_routes():
    """Returns a dictionary of built-in routes."""
    return {
        'two-stop-fixed': Routes(
            routes=[2],
            distance=[[1, 0]],
            traffic=Fixed(1),
            demand_loading=FixedAlternating([[[1, 0], [0, 0]]]),
            demand_unloading=Fixed([[0, 1]]),
        ),
        'two-stop-random': Routes(
            routes=[2],
            distance=[[1, 0]],
            traffic=Fixed(1),
            demand_loading=Pois([[1, 0]]),
            demand_unloading=Pois([[0, 1]]),
        ),
        'ten-stop': Routes(
            routes=[10],
            distance=[[1] * 10],
            traffic=Fixed(1),
            demand_loading=TimeVarPois([[1] * 9 + [0]],
                                       IndicatorKernel(1, 0, 120)),
            demand_unloading=Pois([[0] + [1] * 9]),
        ),
        'b35': Routes(
            routes=[48],
            distance=[[.2]*47 + [0]],
            traffic=TrafficModel(Gamma(4, .25),
                                 SumOf([BetaTimeFunc(8, 5, pdf=True),
                                        BetaTimeFunc(10, 15, pdf=True)]),
                                 daily_func=Beta(6, 4, bias=.5)),
            demand_loading=TimeVarPois([[117] * 47 + [0]],
                                       SumOfDistributionKernel([
                                           BetaTimeFunc(6, 14, area=0.5),
                                           BetaTimeFunc(4, 2, area=0.5),
                                       ])),
            demand_unloading=Pois(([[0] + [117] * 47])),
        ),
        'b35-busy': Routes(
            routes=[48],
            distance=[[.2]*48],
            traffic=TrafficModel(Gamma(4, .25),
                                 SumOf([BetaTimeFunc(8, 5, pdf=True),
                                        BetaTimeFunc(10, 15, pdf=True)]),
                                 daily_func=Beta(6, 4, bias=.5)),
            demand_loading=TimeVarPois([[180] * 47 + [0]],
                                       SumOfDistributionKernel([
                                           BetaTimeFunc(4, 2, area=0.5),
                                           BetaTimeFunc(6, 14, area=0.5),
                                       ])),
            demand_unloading=Pois(([[0] + [180] * 47])),
        ),
    }


def get_builtin_experiments():
    """Returns a dictionary of built-in experiments."""
    routes = get_builtin_routes()
    return {
        'simple': Experiment(
            routes=routes['two-stop-fixed'],
            time_loading=Fixed(1),
            time_unloading=Fixed(1),
            schedule=[[10]],
            headers=Headers.SIMPLE
        ),
        'two-stop': Experiment(
            routes=routes['two-stop-random'],
            time_loading=Fixed(.01),
            time_unloading=Fixed(.1),
            schedule=[[1]],
            headers=Headers.SIMPLE
        ),
        'ten-stop': Experiment(
            routes=routes['ten-stop'],
            time_loading=Pert(1/60, 8/60, 120/60),
            time_unloading=Fixed(.05),
            schedule=[[5, 10, 15, 20, 30]],
            headers=Headers.SIMPLE,
        ),
        'ten-stop-long': Experiment(
            routes=routes['ten-stop'],
            time_loading=Pert(3/60, 15/60, 120/60),
            time_unloading=Fixed(.05),
            schedule=[[5, 10, 15, 20, 30]],
            headers=Headers.SIMPLE,
        ),
        'b35-short': Experiment(
            routes=routes['b35'],
            time_loading=Pert(1/60, 8/60, 120/60, lamb=3,
                              scale=capacity_scale),
            time_unloading=Fixed(.05),
            schedule=b35_schedule(),
            headers=Headers.EXTENDED,
        ),
        'b35-long': Experiment(
            routes=routes['b35'],
            time_loading=Pert(3/60, 15/60, 120/60, lamb=3,
                              scale=capacity_scale),
            time_unloading=Fixed(.05),
            schedule=b35_schedule(),
            headers=Headers.EXTENDED,
        ),
        'b35-short-busy': Experiment(
            routes=routes['b35-busy'],
            time_loading=Pert(1/60, 8/60, 120/60, lamb=3,
                              scale=capacity_scale),
            time_unloading=Fixed(.05),
            schedule=b35_schedule(),
            headers=Headers.EXTENDED,
        ),
        'b35-long-busy': Experiment(
            routes=routes['b35-busy'],
            time_loading=Pert(3/60, 15/60, 120/60, lamb=3,
                              scale=capacity_scale),
            time_unloading=Fixed(.05),
            schedule=b35_schedule(),
            headers=Headers.EXTENDED,
        ),
    }


@auto_repr
class CapacityScale:
    def __init__(self):
        pass

    def __call__(self, t, passengers):
        return t * min(1, 1.1**(passengers - 20))


capacity_scale = CapacityScale()
