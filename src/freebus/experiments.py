"""Experiment parameters."""

from dataclasses import dataclass
from zlib import crc32

from .randomvar import RandomVar, Fixed, FixedAlternating, Pois, Pert, \
    TimeVarPois, IndicatorKernel


class Headers:
    # pylint: disable=too-few-public-methods
    # this class is merely a collection of constant tuples
    """Preset collections of experimental values to record for a
    trial."""
    SIMPLE = ('waiting-time', 'loading-time', 'moving-time',
              'holding-time', 'total-passengers')


@dataclass
class Routes:
    """Parameters for external conditions for a predefined collection of
    bus-stops."""
    routes: list[int]
    distance: list[list[int]]
    traffic: list[list[int]]
    demand_loading: RandomVar
    demand_unloading: RandomVar


@dataclass
class WorldModeling:
    """Parameters for modeling various events."""
    time_loading: RandomVar
    time_unloading: RandomVar


@dataclass
class Schedule:
    """Parameters for bus schedule policies."""
    schedule: list[list[float]]


class Experiment:
    """Object containing experimental parameters."""
    def __init__(self,
                 routes,
                 time_loading, time_unloading,
                 schedule,
                 headers):
    # def __init__(self, routes, distance, traffic, demand_loading,
    #              demand_unloading, time_loading, time_unloading,
    #              schedule, headers):
        self._routes = routes
        # self.distance = distance
        # self.traffic = traffic
        # self.demand_loading = demand_loading
        # self.demand_unloading = demand_unloading
        self.time_loading = time_loading
        self.time_unloading = time_unloading
        self.schedule = schedule
        self.headers = headers

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

    def __repr__(self):
        return (
            f'{type(self).__name__}('
            f'{self.routes}, {self.distance}, '
            f'{self.schedule}, {self.headers}, '
            f'traffic={self.traffic}, demand_loading={self.demand_loading}, '
            f'demand_unloading={self.demand_unloading}, '
            f'time_loading={self.time_loading})')


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
            distance=[[.2]*48],
            traffic=Fixed(.75),
            demand_loading=TimeVarPois([[.07] * 47 + [0]],
                                       IndicatorKernel(1, 0, 120)),
            demand_unloading=Pois(([[0] + [.07] * 47])),
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
            time_loading=Pert(1/60, 8/60, 120/60),
            time_unloading=Fixed(.05),
            schedule=[
                [30, 80, 130, 180, 225, 265, 295, 318, 339, 355, 375,
                 382, 389, 396, 403, 411, 418, 426, 433, 441, 448, 456,
                 463, 471, 478, 486, 493, 501, 509, 517, 526, 538, 550,
                 562, 574, 586, 598, 610, 622, 634, 646, 658, 670, 682,
                 694, 706, 718, 730, 742, 754, 764, 774, 784, 794, 804,
                 814, 824, 834, 844, 854, 864, 874, 881, 889, 896, 904,
                 911, 919, 926, 934, 942, 950, 958, 966, 974, 982, 990,
                 998, 1006, 1014, 1022, 1029, 1037, 1044, 1051, 1058,
                 1065, 1072, 1079, 1086, 1094, 1103, 1112, 1121, 1130,
                 1139, 1148, 1157, 1166, 1175, 1185, 1197, 1209, 1221,
                 1233, 1245, 1257, 1269, 1281, 1296, 1315, 1331, 1343,
                 1355, 1367, 1379, 1392, 1405, 1420, 1436]
            ],
            headers=Headers.SIMPLE,
        ),
    }
