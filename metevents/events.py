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
    def from_station(cls, station_id, start, end):
        raise NotImplementedError('Not implemented')


class StormEvents(BaseEvents):

    def find(self, instant_mass_to_start=0.1, min_storm_total=0.5, hours_to_stop=24,
             max_storm_hours=336):
        """
        Find all the storms that are initiated by a mass greater than the
        instant_mass_to_start and receive less than that threshold for at
        least hours_to_stop to end it. Storm delineation is further bounded by
        min_storm_total and max_storm_hours.

        Args:
            instant_mass_to_start: mass per time step to consider the beginning of a
                storm
            min_storm_total: Total storm mass to be considered a complete storm
            hours_to_stop: minimum hours of mass less than instant threshold to
                end a storm
            max_storm_hours: Maximum hours a storm can.
        """
        # group main condition by time
        ind = self.data >= instant_mass_to_start
        groups, _ = self.group_condition_by_time(ind)

        freq = determine_freq(ind)
        tstep = pd.to_timedelta(to_offset(freq))
        dt = timedelta(hours=hours_to_stop)
        max_storm = timedelta(hours=max_storm_hours)

        group_list = sorted(list(groups.items()))
        N_groups = len(group_list)

        # Evaluate each group of mass conditions against the timing
        for i, (event_id, curr_group) in enumerate(group_list):
            curr_start = curr_group.min()
            curr_stop = curr_group.max()
            if i == 0:
                start = curr_start

            # Grab next
            nx_idx = i + 1
            if nx_idx < N_groups:
                next_group = group_list[nx_idx][1]
                next_start = next_group.min()

            else:
                next_start = curr_stop
            # track storm total and no_precip_d
            total = self.data.loc[start:curr_stop].sum()
            duration = curr_stop - start

            # Has there been enough hours without mass
            enough_hours_wo_precip = (next_start - curr_stop) > dt
            # Has storm gone on too long
            storm_duration_too_long = duration > max_storm
            # Has enough mass accumulated to be considered a storm
            enough_storm_mass = total >= min_storm_total
            base_condition = (enough_hours_wo_precip or storm_duration_too_long)
            condition = (base_condition and enough_storm_mass)

            if condition or nx_idx == N_groups:
                # Watch out for beginning
                start = start - tstep if start != self.data.index[0] else start

                event = CumulativePeriod(self.data.loc[start:curr_stop])
                self._events.append(event)
                # Update start for the next storm
                start = next_start

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

        if df is None:
            raise ValueError(f'The combination of pulling precip from {station_id} '
                             f'during {start}-{stop} produced no data. Check station '
                             f'is real and has precip data between specified dates.')
        else:
            df = df.reset_index().set_index('datetime')

        return cls(df[variable.name].diff())
