"""Experiment parameters."""

from zlib import crc32

class Experiment:
    """Object containing experimental parameters."""
    def __init__(self, routes, distance, traffic, demand_loading,
            demand_unloading, time_loading, time_unloading,
            schedule, headers):
        self.routes = routes
        self.distance = distance
        self.traffic = traffic
        self.demand_loading = demand_loading
        self.demand_unloading = demand_unloading
        self.time_loading = time_loading
        self.time_unloading = time_unloading
        self.schedule = schedule
        self.headers = headers

    def checksum(self):
        """Returns a crc32 checksum of the experimental parameters as a
        hexadecimal string."""
        return hex(crc32(str(self).encode('utf8')))[2:]

    def __repr__(self):
        return (f'{type(self).__name__}('
        f'{self.routes}, {self.distance}, {self.schedule}, {self.headers}, '
        f'traffic={self.traffic}, demand_loading={self.demand_loading}, '
        f'demand_unloading={self.demand_unloading}, time_loading={self.time_loading})')
