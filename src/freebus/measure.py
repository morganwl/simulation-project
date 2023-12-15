"""Generate measurements based on events."""

from collections import defaultdict

import numpy as np


def measure(events, headers):
    """Returns an array of variables, measured from a list of events."""
    rv = {}
    total_passengers = max(measure_passengers(events),
                           np.finfo(np.float64).tiny)
    for h in headers:
        if h == 'total-passengers':
            rv[h] = total_passengers
            continue
        rv[h] = rv_handlers[h](events, total_passengers)
    return np.fromiter((rv[h] for h in headers), dtype=np.float64,
                       count=len(headers))


def measure_bus_load_median(events, _):
    """Measure the median number of passenger per segment (stop to
    stop) of bus travel."""
    return np.quantile(np.array([e.passengers for e in events
                                 if e.etype == 'depart']),
                       .5)


def measure_bus_load_extreme(events, _, percentile=.99):
    """Measure an upper percentile number of passengers per segment."""
    return np.quantile(np.array([e.passengers for e in events
                                 if e.etype == 'depart']),
                       percentile)


def measure_last_event(events, _):
    """Measures the time of the last event in the experiment."""
    return max(e.time for e in events)


def measure_waiting(events, passengers):
    """Measures the mean waiting time."""
    return sum(e.waiting * e.dur for e in events) / passengers


def measure_loading(events, passengers):
    """Returns the mean loading and unloading time per passenger."""
    return (sum(e.passengers * e.dur for e in events
                if e.etype in ['load', 'unload'])
            / passengers)


def measure_moving(events, passengers):
    """Updates the measurement of moving rv based on an event."""
    return sum(e.passengers * e.dur for e in events
               if e.etype == 'depart') / passengers


def measure_holding(events, passengers):
    """Measures the time spent holding."""
    return sum(e.passengers * e.dur for e in events
               if e.etype == 'hold') / passengers


def measure_passengers(events):
    """Measures the number of passengers."""
    passengers = 0
    buses = defaultdict(int)
    for e in events:
        if e.etype == 'load':
            passengers += e.passengers - buses[(e.route, e.busid)]
            buses[(e.route, e.busid)] = e.passengers
        if e.etype == 'unload':
            buses[(e.route, e.busid)] = e.passengers
    return passengers


def measure_pass_range(events, start, stop):
    """Measures the number of passengers within a particular time
    range."""
    passengers = 0
    buses = {}
    for e in events:
        if e.etype == 'load' and start <= e.time < stop:
            passengers += e.passengers - buses[e.route, e.busid].passengers
        buses[e.route, e.busid] = e
    return passengers


rv_handlers = (
    {
        'waiting-time': measure_waiting,
        'loading-time': measure_loading,
        'moving-time': measure_moving,
        'holding-time': measure_holding,
        'total-passengers': measure_passengers,
        'last-event': measure_last_event,
        'median-load': measure_bus_load_median,
        'extreme-load': measure_bus_load_extreme,
    }
    |
    {f'passengers-{i}': (lambda x, _, i=i: measure_pass_range(x, 60 * i,
                                                              60 * (i + 1)))
     for i in range(24)}
)