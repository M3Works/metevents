import numpy as np
from .periods import CumulativePeriod


class BaseEvents:
    def __init__(self, data):
        self._events = []
        self.data = data

    @property
    def events(self):
        return self._events

    @property
    def N(self):
        return len(self.events)

    def find(self, *args, **kwargs):
        """
        Function to be defined for specific events in timeseries data. Performs
        the actual detection of the events. Should assign self._events
        """
        raise NotImplementedError("Find function not implemented.")

    @classmethod
    def get_start_stop(cls, ind):
        """
        Given a boolean index, find the start stop of True sections
        """
        ind_sum = ind.eq(False).cumsum()
        start_stops = ind_sum.loc[ind.eq(True)].groupby(ind_sum).apply(lambda d: [d.index.min(), d.index.max()])
        return start_stops

    @classmethod
    def from_station(cls):
        raise NotImplementedError('Not implemented')


class StormEvents(BaseEvents):
    @classmethod
    def from_series(cls, mass_series):
        """
        Take in a pandas timeseries of mass
        """
        return StormEvents(mass_series)

    def find(self, mass_to_start=0.1, hours_to_stop=24):
        """
        Find all the storms that are initiated by a mass greater than the
        mass_to_start and receive less than that threshold for at
        least hours_to_stop
        """
        ind = self.data >= mass_to_start
        start_stops = self.get_start_stop(ind)
        for (start, stop) in start_stops:
            evt = CumulativePeriod(self.data.loc[start:stop])
            self._events.append(evt)
