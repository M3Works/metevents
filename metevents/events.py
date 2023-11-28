import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from metloom.pointdata import CDECPointData, SnotelPointData, MesowestPointData
from pandas.tseries.frequencies import to_offset
from scipy.signal import find_peaks

# local imports
from metevents.periods import CumulativePeriod, BaseTimePeriod
from metevents.utilities import determine_freq


LOG = logging.getLogger(__name__)


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

        # Isolate the ind_sum by positions that are
        # True and group them together
        time_groups = ind_sum.loc[ind.eq(True)].groupby(ind_sum)
        groups = time_groups.groups
        return groups, ind_sum

    @classmethod
    def from_station(cls, station_id, start, end):
        raise NotImplementedError('Not implemented')


class StormEvents(BaseEvents):

    def find(self, instant_mass_to_start=0.1, min_storm_total=0.5,
             hours_to_stop=24, max_storm_hours=336):
        """
        Find all the storms that are initiated by a mass greater than the
        instant_mass_to_start and receive less than that threshold for at
        least hours_to_stop to end it. Storm delineation is further bounded by
        min_storm_total and max_storm_hours.

        Args:
            instant_mass_to_start: mass per time step to consider the
                beginning of a storm
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
            base_condition = (
                enough_hours_wo_precip or storm_duration_too_long
            )
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
            source: Network/datasource to search for data options:
                NRCS, mesowest, CDEC
            station_name: String name of the station to pass to pointdata
        """
        pnt = None
        pnt_classes = [SnotelPointData, CDECPointData, MesowestPointData]
        for STATION_CLASS in pnt_classes:
            if STATION_CLASS.DATASOURCE.lower() == source.lower():
                pnt = STATION_CLASS(station_id, station_name)
                break

        if pnt is None:
            raise ValueError(
                f'Datasource {source} is invalid. Use '
                f'{", ".join([c.DATASOURCE for c in pnt_classes])}'
            )

        # Pull data
        variable = pnt.ALLOWED_VARIABLES.PRECIPITATIONACCUM

        df = pnt.get_daily_data(start, stop, [variable])

        if df is None:
            raise ValueError(
                f'The combination of pulling precip from {station_id} '
                f'during {start}-{stop} produced no data. Check station '
                f'is real and has precip data between specified dates.'
            )
        else:
            df = df.reset_index().set_index('datetime')

        return cls(df[variable.name].diff())


class SpikeValleyEvent(BaseEvents):

    def find(
            self, height=None, threshold=None, prominence=100.0,
            width=None
    ):
        """
        Find instances of spikes or valleys within a timeseries

        Args:
            height: Required height of peaks
            threshold: Required relative height to neighboring peaks
            prominence: Required prominence of peaks
            width: required width. Default is a min of 0 and max of 3 (0,3)

        """
        ind = self.detect_spikes_using_find_peaks(
            self.data, height=height, threshold=threshold,
            prominence=prominence, width=width
        )
        # Group the events
        groups, _ = self.group_condition_by_time(ind)
        group_list = sorted(list(groups.items()))

        # Build the list of events
        for i, (event_id, curr_group) in enumerate(group_list):
            curr_start = curr_group.min()
            curr_stop = curr_group.max()
            event = BaseTimePeriod(self.data.loc[curr_start:curr_stop])
            self._events.append(event)

    @staticmethod
    def detect_spikes_using_find_peaks(
            series, height=None, threshold=None, prominence=100.0,
            width=None
    ):
        """
        Detect spikes in time series data using the scipy find_peaks function
        https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.signal.find_peaks.html

        Args:
            series: A pandas Series representing time series data.
            height: Required height of peaks
            threshold: Required relative height to neighboring peaks
            prominence: Required prominence of peaks
            width: required width. Default is a min of 0 and max of 3 (0,3)

        Returns:

        """
        width = width or (0, 3)

        # find peaks
        peaks, peak_info = find_peaks(
            series, height=height, threshold=threshold, prominence=prominence,
            width=width
        )
        # get the index span
        peak_width_values = peak_info["widths"]

        # find valleys
        valleys, valley_info = find_peaks(
            # Data gets flipped around y axis
            series * -1.0,
            height=height, threshold=threshold, prominence=prominence,
            width=width
        )
        valley_width_values = valley_info["widths"]

        spike_index = pd.Series(index=series.index, data=[False] * len(series))
        # Set true for the width of the peak surrounding the center
        for p, w in zip(peaks, peak_width_values):
            p1 = int(p - w)
            p2 = int(p + w) + 1
            spike_index.iloc[p1:p2] = True
        for p, w in zip(valleys, valley_width_values):
            p1 = int(p - w)
            p2 = int(p + w) + 1
            spike_index.iloc[p1:p2] = True
        return spike_index


class DataGapEvent(BaseEvents):

    def find(self, min_len=3, expected_frequency="1D"):
        """
        Find instances of data gaps

        Args:
            min_len: minimum length of a gap
            expected_frequency: expected frequency of timeseries

        """
        # find all nan indices
        ind = pd.isna(self.data)

        # Ensure the dataframe is sorted by index
        self.data = self.data.sort_index()

        # Calculate differences between consecutive timestamps
        differences = self.data.index.to_series().diff()

        # Assume a daily frequency for this example
        expected_difference = pd.Timedelta(expected_frequency)

        # Group the nan data events
        groups, _ = self.group_condition_by_time(ind)

        # Identify gap start points
        # gap_starts = self.data.index[differences > expected_difference]
        gaps = differences[differences > expected_difference].dropna()
        for idg, gap in zip(gaps.index, gaps):
            # TODO: this logic makes missing 4 days into a 6 day gap
            gap_iloc = ind.index.get_loc(idg)
            # Create a group of the missing indices
            groups[gap_iloc] = pd.DatetimeIndex(
                [idg - gap, idg]
            )

        # sort the group list
        group_list = sorted(list(groups.items()))

        # Build the list of events
        for event_id, curr_group in group_list:
            curr_start = curr_group.min()
            curr_stop = curr_group.max()
            event = BaseTimePeriod(self.data.loc[curr_start:curr_stop])
            # only keep events that are longer than what is configured
            if event.duration >= min_len * expected_difference:
                self._events.append(event)


class FlatLineEvent(BaseEvents):

    def find(self, min_len=5, slope_thresh=0.0):
        """
        Find instances of flat-lined data

        Args:
            min_len: minimum length of a flat value
            slope_thresh: slope threshold for flatline. Anything absolute
                value of slope <=slope thresh will be flagged

        """
        # find the slope
        diff = self.data.diff()
        # find the absolute slope within our threshold
        ind = np.abs(diff) <= slope_thresh

        # Group the nan data events
        groups, _ = self.group_condition_by_time(ind)
        # sort the group list
        group_list = sorted(list(groups.items()))

        # Build the list of events
        for event_id, curr_group in group_list:
            curr_start = curr_group.min()
            curr_stop = curr_group.max()
            event = BaseTimePeriod(self.data.loc[curr_start:curr_stop])
            # only keep events that are longer than what is configured
            if len(event.data) >= min_len:
                self._events.append(event)


class ExtremeValueEvent(BaseEvents):

    def find(self, expected_max=600.0, expected_min=0.0):
        """
        Find events where the values are outside of the expected range

        Args:
            expected_max: Maximum expected value in the data
            expected_min: minimum expected value in the data

        """
        # Indices where data is outside of expected range
        ind = (self.data > expected_max) | (self.data < expected_min)

        # Group the nan data events
        groups, _ = self.group_condition_by_time(ind)
        # sort the group list
        group_list = sorted(list(groups.items()))

        # Build the list of events
        for event_id, curr_group in group_list:
            curr_start = curr_group.min()
            curr_stop = curr_group.max()
            event = BaseTimePeriod(self.data.loc[curr_start:curr_stop])
            # store the events
            self._events.append(event)


class ExtremeChangeEvent(BaseEvents):
    """
    Find where the slope of the data is larger than expected over a certain
    period of time
    """

    def find(
            self, min_len=1, positive_slope_thresh=None,
            negative_slope_thresh=-3.0
    ):
        """
        Find instances of excessive rate of change

        Args:
            min_len: minimum length of the event
            positive_slope_thresh: Anything absolute
                value of slope >=slope thresh will be flagged
            negative_slope_thresh: slope threshold for flatline. Anything absolute
                value of slope >=slope thresh will be flagged

        """

        if positive_slope_thresh is None and negative_slope_thresh is None:
            raise ValueError("One slope threshold must be provided")

        # find the slope
        diff = self.data.diff()

        ind_pos = pd.Series([False] * len(diff), index=diff.index)
        ind_neg = pd.Series([False] * len(diff), index=diff.index)
        if positive_slope_thresh is not None:
            ind_pos = diff >= positive_slope_thresh
        if negative_slope_thresh is not None:
            ind_neg = diff <= negative_slope_thresh

        # join the index
        ind = ind_pos | ind_neg

        # Group the nan data events
        groups, _ = self.group_condition_by_time(ind)
        # sort the group list
        group_list = sorted(list(groups.items()))

        # Build the list of events
        for event_id, curr_group in group_list:
            curr_start = curr_group.min()
            curr_stop = curr_group.max()
            event = BaseTimePeriod(self.data.loc[curr_start:curr_stop])
            # only keep events that are longer than what is configured
            if len(event.data) >= min_len:
                self._events.append(event)
