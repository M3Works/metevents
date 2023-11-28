from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DatetimeIndex

from metevents.events import (
    StormEvents, SpikeValleyEvent, DataGapEvent, FlatLineEvent,
    ExtremeValueEvent, ExtremeChangeEvent
)


@pytest.fixture()
def series(data):
    index = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(len(data))]
    return pd.Series(data, index=DatetimeIndex(index, freq='D'))


class TestStormEvents:
    @pytest.fixture()
    def storms(self, series, data):
        yield StormEvents(series)

    @pytest.mark.parametrize('data, start_mass, stop_hours, total_mass, max_hours,'
                             'n_storms', [
                                 ([0, 1, 1, 0, 0, 1, 1], 0.1, 24, 1, 300, 2),
                                 # Test stopping hours
                                 ([0, 0.1, 0.1, 0, 0.1, 0.1], 0.1, 48, 0.1, 300, 1),
                                 # Test minimum storm total
                                 ([0.1, 0, 0.1, 0.1], 0.1, 24, 0.2, 300, 1),
                                 # Test max storm hours
                                 ([0, 0.1, 0, 0.1, 0.1, 0], 0.1, 24, 0.1, 24, 2),

                             ])
    def test_storm_events_N(self, storms, data, start_mass, stop_hours,
                            total_mass, max_hours, n_storms):
        """
        Test the number of storms identified by varying input data
        and thresholds.
        """
        storms.find(instant_mass_to_start=start_mass,
                    hours_to_stop=stop_hours,
                    min_storm_total=total_mass,
                    max_storm_hours=max_hours)
        assert storms.N == n_storms

    @pytest.mark.parametrize('data, mass, hours, totals', [
        # Two storms
        ([0, 1, 1, 0, 0, 1, 1], 0.1, 24, [2, 2]),
        # Same data but 1 storm using different criteria
        ([0, 1, 1, 0, 0, 1, 1], 0.1, 72, [4]),

    ])
    def test_storm_events_total(self, storms, data, mass, hours, totals):
        """
        Test the number of storms identified by varying input data
        and thresholds.
        """
        storms.find(instant_mass_to_start=mass, hours_to_stop=hours)
        assert [event.total for event in storms.events] == totals

    @pytest.mark.parametrize('data, mass, hours, durations', [
        # Two storms with clear delineation
        ([0, 1, 1, 0, 0, 1, 1], 0.1, 24, [2, 2]),
        # No clear beginning
        ([0.2, 1, 0, 1, 0.2, 1], 0.1, 24, [1, 3]),
        ([1, 1, 1, 1], 0.1, 24, [3]),

    ])
    def test_storm_events_duration(self, storms, data, mass, hours, durations):
        """
        Test the number of storms identified by varying input data
        and thresholds.
        """
        storms.find(instant_mass_to_start=mass, hours_to_stop=hours)
        assert [event.duration for event in storms.events] == \
               [timedelta(days=t) for t in durations]

    @pytest.mark.parametrize('station_id, start, stop, source, mass, hours, n_storms', [
        ('TUM', datetime(2021, 12, 1), datetime(2022, 1, 15), 'CDEC', 0.1, 48, 5),
        ('637:ID:SNTL', datetime(2022, 12, 1), datetime(2022, 12, 15),
         'NRCS', 0.1, 48, 2)

    ])
    def test_storm_events_from_station(self, station_id, start, stop, source, mass,
                                       hours, n_storms):
        """
        Test the number of storms identified by varying input data and thresholds.
        """
        storms = StormEvents.from_station(station_id, start, stop, source=source)
        storms.find(instant_mass_to_start=mass, hours_to_stop=hours,
                    min_storm_total=0.2)
        assert storms.N == n_storms


class TestSpikeValleyEvent:
    DATA_DIR = Path(__file__).parent.joinpath("data/mocks")

    @pytest.fixture(scope="class")
    def series(self):
        df = pd.read_csv(
            self.DATA_DIR.joinpath("flv.csv"), parse_dates=["datetime"],
            index_col="datetime"
        )
        return df["SNOWDEPTH"]

    @pytest.fixture(scope="class")
    def events(self, series):
        yield SpikeValleyEvent(series)

    @pytest.fixture(scope="class")
    def found_events(self, events):
        events.find()
        yield events

    def test_number_of_events(self, found_events):
        assert found_events.N == 11

    @pytest.mark.parametrize(
        "idx, start_date", [
            (0, '2022-11-01T08:00:00+00:00'),
            (1, '2022-11-11T08:00:00+00:00'),
            (2, '2022-11-30T08:00:00+00:00'),
            (3, '2022-12-29T08:00:00+00:00'),
            (4, '2023-01-04T08:00:00+00:00'),
            (5, '2023-01-15T08:00:00+00:00'),
            (6, '2023-01-28T08:00:00+00:00'),
            (7, '2023-02-04T08:00:00+00:00'),
            (8, '2023-02-12T08:00:00+00:00'),
            (9, '2023-02-19T08:00:00+00:00'),
            (10, '2023-04-23T08:00:00+00:00')
        ]
    )
    def test_start_dates(self, found_events, idx, start_date):
        event = found_events.events[idx]
        assert event.start == pd.to_datetime(start_date)

    @pytest.mark.parametrize(
        "idx, stop_date", [
            (0, '2022-11-04T08:00:00+00:00'),
            (1, '2022-11-14T08:00:00+00:00'),
            (2, '2022-12-14T08:00:00+00:00'),
            (3, '2023-01-02T08:00:00+00:00'),
            (4, '2023-01-07T08:00:00+00:00'),
            (5, '2023-01-20T08:00:00+00:00'),
            (6, '2023-01-31T08:00:00+00:00'),
            (7, '2023-02-07T08:00:00+00:00'),
            (8, '2023-02-17T08:00:00+00:00'),
            (9, '2023-03-05T08:00:00+00:00'),
            (10, '2023-04-26T08:00:00+00:00')
        ]
    )
    def test_stop_dates(self, found_events, idx, stop_date):
        event = found_events.events[idx]
        assert event.stop == pd.to_datetime(stop_date)

    @pytest.mark.parametrize(
        "idx, duration", [
            (0, '3 days 00:00:00'),
            (1, '3 days 00:00:00'),
            (2, '14 days 00:00:00'),
            (3, '4 days 00:00:00'),
            (4, '3 days 00:00:00'),
            (5, '5 days 00:00:00'),
            (6, '3 days 00:00:00'),
            (7, '3 days 00:00:00'),
            (8, '5 days 00:00:00'),
            (9, '14 days 00:00:00'),
            (10, '3 days 00:00:00')
        ]
    )
    def test_start_duration(self, found_events, idx, duration):
        event = found_events.events[idx]
        assert event.duration == pd.to_timedelta(duration)


class TestDataGapEvent:

    @pytest.fixture(scope="class")
    def gap_series(self):
        data = np.array(range(100)).astype("float32")
        index = [datetime(2023, 1, 1) + timedelta(days=i) for i in
                 range(len(data))]
        # Set nans that we will drop
        data[10:15] = np.nan
        data[40:45] = np.nan
        # gap not big enough to flag
        data[50:51] = np.nan
        series = pd.Series(data, index=DatetimeIndex(index, freq='D'))
        # Drop na to create time gaps
        series = series.dropna()
        # create nan that should be flagged
        series.iloc[60:65] = np.nan
        return series

    @pytest.fixture(scope="class")
    def events(self, gap_series):
        yield DataGapEvent(gap_series)

    @pytest.fixture(scope="class")
    def found_events(self, events):
        events.find(min_len=3, expected_frequency="1D")
        yield events

    def test_number_of_events(self, found_events):
        assert found_events.N == 3

    @pytest.mark.parametrize(
        "idx, start_date", [
            (0, "2023-01-10"),
            (1, "2023-02-09"),
            (2, "2023-03-13"),
        ]
    )
    def test_start_dates(self, found_events, idx, start_date):
        event = found_events.events[idx]
        assert event.start == pd.to_datetime(start_date)

    @pytest.mark.parametrize(
        "idx, duration", [
            (0, "6 days"),
            (1, "6 days"),
            (2, "4 days"),
        ]
    )
    def test_start_duration(self, found_events, idx, duration):
        event = found_events.events[idx]
        assert event.duration == pd.to_timedelta(duration)


class TestFlatlineEvent:

    @pytest.fixture(scope="class")
    def flat_series(self):
        data = np.array(range(100)).astype("float32")
        index = [datetime(2023, 1, 1) + timedelta(days=i) for i in
                 range(len(data))]
        # Set flatlines
        data[10:18] = 10.0
        data[40:48] = 40.0
        # not long enough to flag
        data[50:54] = 50.0
        series = pd.Series(data, index=DatetimeIndex(index, freq='D'))
        return series

    @pytest.fixture(scope="class")
    def events(self, flat_series):
        yield FlatLineEvent(flat_series)

    @pytest.fixture(scope="class")
    def found_events(self, events):
        events.find(min_len=5, slope_thresh=0.0)
        yield events

    def test_number_of_events(self, found_events):
        assert found_events.N == 2

    @pytest.mark.parametrize(
        "idx, start_date", [
            (0, "2023-01-12"),
            (1, "2023-02-11"),
        ]
    )
    def test_start_dates(self, found_events, idx, start_date):
        event = found_events.events[idx]
        assert event.start == pd.to_datetime(start_date)

    @pytest.mark.parametrize(
        "idx, stop_date", [
            (0, "2023-01-18"),
            (1, "2023-02-17"),
        ]
    )
    def test_stop_dates(self, found_events, idx, stop_date):
        event = found_events.events[idx]
        assert event.stop == pd.to_datetime(stop_date)

    @pytest.mark.parametrize(
        "idx, duration", [
            (0, "6 days"),
            (1, "6 days"),
        ]
    )
    def test_start_duration(self, found_events, idx, duration):
        event = found_events.events[idx]
        assert event.duration == pd.to_timedelta(duration)


class TestExtremeValueEvent:

    @pytest.fixture(scope="class")
    def series(self):
        data = np.array(range(100)).astype("float32")
        index = [datetime(2023, 1, 1) + timedelta(days=i) for i in
                 range(len(data))]
        # Set extreme values
        data[10:15] = 700.0
        data[40:48] = -1.0
        data[50:54] = 601.0
        series = pd.Series(data, index=DatetimeIndex(index, freq='D'))
        return series

    @pytest.fixture(scope="class")
    def events(self, series):
        yield ExtremeValueEvent(series)

    @pytest.fixture(scope="class")
    def found_events(self, events):
        events.find(expected_max=600.0, expected_min=0.0)
        yield events

    def test_number_of_events(self, found_events):
        assert found_events.N == 3

    @pytest.mark.parametrize(
        "idx, start_date", [
            (0, "2023-01-11"),
            (1, "2023-02-10"),
            (2, "2023-02-20"),
        ]
    )
    def test_start_dates(self, found_events, idx, start_date):
        event = found_events.events[idx]
        assert event.start == pd.to_datetime(start_date)

    @pytest.mark.parametrize(
        "idx, stop_date", [
            (0, "2023-01-15"),
            (1, "2023-02-17"),
            (2, "2023-02-23"),
        ]
    )
    def test_stop_dates(self, found_events, idx, stop_date):
        event = found_events.events[idx]
        assert event.stop == pd.to_datetime(stop_date)

    @pytest.mark.parametrize(
        "idx, duration", [
            (0, "4 days"),
            (1, "7 days"),
            (2, "3 days"),
        ]
    )
    def test_start_duration(self, found_events, idx, duration):
        event = found_events.events[idx]
        assert event.duration == pd.to_timedelta(duration)


class TestExtremeChangeEvent:
    @pytest.fixture(scope="class")
    def series(self):
        data = np.array(range(100)).astype("float32")
        index = [datetime(2023, 1, 1) + timedelta(days=i) for i in
                 range(len(data))]
        # Set extreme values
        data[10:15] = 700.0
        series = pd.Series(data, index=DatetimeIndex(index, freq='D'))
        return series

    @pytest.fixture(scope="class")
    def events(self, series):
        yield ExtremeChangeEvent(series)

    @pytest.fixture(scope="class")
    def found_events(self, events):
        events.find(
            min_len=1, positive_slope_thresh=100,
            negative_slope_thresh=-100.0)
        yield events

    def test_number_of_events(self, found_events):
        assert found_events.N == 2

    @pytest.mark.parametrize(
        "idx, start_date", [
            (0, "2023-01-11"),
            (1, "2023-01-16"),
        ]
    )
    def test_start_dates(self, found_events, idx, start_date):
        event = found_events.events[idx]
        assert event.start == pd.to_datetime(start_date)

    @pytest.mark.parametrize(
        "idx, stop_date", [
            (0, "2023-01-11"),
            (1, "2023-01-16"),
        ]
    )
    def test_stop_dates(self, found_events, idx, stop_date):
        event = found_events.events[idx]
        assert event.stop == pd.to_datetime(stop_date)

    @pytest.mark.parametrize(
        "idx, duration", [
            (0, "0 days"),
            (1, "0 days"),
        ]
    )
    def test_start_duration(self, found_events, idx, duration):
        event = found_events.events[idx]
        assert event.duration == pd.to_timedelta(duration)
