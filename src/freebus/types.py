"""Shared data types."""

from dataclasses import dataclass

@dataclass(slots=True, order=True, frozen=True)
class Event:
    time: float
    dur: float
    etype: str
    route: int
    stop: int
    busid: int
    passengers: int
