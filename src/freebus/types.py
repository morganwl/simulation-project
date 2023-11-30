"""Shared data types."""

from collections import namedtuple

Event = namedtuple('Event', ['time', 'dur', 'etype', 'route', 'stop', 'busid', 'passengers'])
