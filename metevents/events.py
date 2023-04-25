import numpy as np
from .periods import CumulativePeriod
import pandas as pd
from datetime import timedelta
from metloom.pointdata import CDECPointData, SnotelPointData, MesowestPointData
from pandas.tseries.frequencies import to_offset
from .utilities import determine_freq


class BaseEvents:
    def __init__(self, data):
        self._events = []
        self.data = data
        self._groups = []
        self._group_ids = None

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
        raise NotImplementedError("find function not implemented.")

    @staticmethod
    def group_condition_by_time(ind):
        ind_sum = ind.eq(False).cumsum()

        # Isolate the ind_sum by positions that are True and group them together
        time_groups = ind_sum.loc[ind.eq(True)].groupby(ind_sum)
        groups = time_groups.groups
        return groups, ind_sum

    @classmethod
    def get_timedelta(cls, ind):
        """
        Determine the timedelta of each continuous section of boolean thats
        true. NaT is used when the bool is False
        """
        # group together the continuous true conditions
        groups, ind_sum = cls.group_condition_by_time(ind)
        nat = np.datetime64('NaT')
        freq = determine_freq(ind)
        add_one = pd.to_timedelta(to_offset(freq))
        # Always add one since we want to include that last timestep
        result = ind_sum.apply(
            lambda sum_id: groups[sum_id].max() - groups[sum_id].min() + add_one
            if sum_id in groups else nat)
        return result

    @classmethod
    def from_station(cls, station_id, start, end):
        raise NotImplementedError('Not implemented')


class StormEvents(BaseEvents):

    def find(self, mass_to_start=0.1, hours_to_stop=24):
        """
        Find all the storms that are initiated by a mass greater than the
        mass_to_start and receive less than that threshold for at
        least hours_to_stop
        """
        ind = self.data >= mass_to_start
        delta = self.get_timedelta(ind)
        ind = ind & (delta >= timedelta(hours=hours_to_stop))
        groups, _ = self.group_condition_by_time(ind)

        for event_id, time_range in groups.items():
            start = time_range.min()
            stop = time_range.max()
            evt = CumulativePeriod(self.data.loc[start:stop])
            self._events.append(evt)

    @classmethod
    def from_station(cls, station_id, start, stop, station_name='unknown',
                     source='NRCS'):
        """

        Form storm analysis from metloom

        Args:
            station_id: string id of the station of interest
            start: Datetime object when to start looking for data
            stop: Datetime object when to stop looking for data
            source: Network/datasource to search for data options: NRCS, mesowest, CDEC
            station_name: String name of the station to pass to pointdata
        """
        pnt = None
        pnt_classes = [SnotelPointData, CDECPointData, MesowestPointData]
        for STATION_CLASS in pnt_classes:
            if STATION_CLASS.DATASOURCE.lower() == source.lower():
                pnt = STATION_CLASS(station_id, station_name)
                break

        if pnt is None:
            raise ValueError(f'Datasource {source} is invalid. Use '
                             f'{", ".join([c.DATASOURCE for c in pnt_classes])}')

        # Pull data
        variable = pnt.ALLOWED_VARIABLES.PRECIPITATIONACCUM

        df = pnt.get_daily_data(start, stop, [variable])
        df = df.reset_index().set_index('datetime')

        if df is None:
            raise ValueError(f'The combination of pulling precip from {station_id} '
                             f'during {start}-{stop} produced no data. Check station '
                             f'is real and has precip data between specified dates.')

        return cls(df[variable.name].diff())
