from datetime import datetime, timedelta
import pandas as pd
import pytest
from pandas import DatetimeIndex

from metevents.events import StormEvents
from metevents.events import OutlierEvents
import numpy as np


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


class TestOutlierEvents:
    @pytest.fixture()
    def outlier_storms(self, series, data):
        yield OutlierEvents(series)

    @pytest.mark.parametrize('data, outliers_value', [
        ([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 2, 3, 3, 3, 3, 23, 4, 42, 2, 2, -40], [42, -40])
    ])
    def test_outliers_value(self, outlier_storms, data, outliers_value):
        outlier_storms.find()
        assert np.all(outlier_storms.values == outliers_value)

    @pytest.mark.parametrize('data, outliers_date', [
        ([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 2, 3, 3, 3, 3, 23, 4, 42, 2, 2, -40],
         [datetime(2023, 1, 26), datetime(2023, 1, 29)])
    ])
    def test_outliers_date(self, outlier_storms, data, outliers_date):
        outlier_storms.find()
        assert np.all(outlier_storms.dates == outliers_date)

    @pytest.mark.parametrize('data', [
        ([2, 2, 2, 2])
    ])
    def test_length(self, outlier_storms, data):
        with pytest.raises(ValueError,
                           match='Data length must be greater than 15 '
                                 'for outlier calculation.'):
            outlier_storms.find()

    @pytest.mark.parametrize('station_id, start, stop, source, out_value', [
        ('TUM', datetime(2021, 10, 1), datetime(2022, 9, 30), 'CDEC',
         [3.34, 2.55, 2.43, 1.54, 1.14])
    ])
    def test_station_outlier_value(self, station_id, start, stop, source, out_value):
        outlier_storms = OutlierEvents.from_station(
            station_id=station_id,
            start=start, stop=stop, source=source
        )

        outlier_storms.find()
        tolerance = 1e-10
        results = outlier_storms.values
        expect_value = pytest.approx(out_value, rel=tolerance, abs=tolerance)
        assert np.all(results == expect_value)

    @pytest.mark.parametrize('station_id, start, stop, source, out_date', [
        ('TUM', datetime(2021, 10, 1), datetime(2022, 9, 30), 'CDEC',
         DatetimeIndex(['2021-10-24 08:00:00+00:00', '2021-10-25 08:00:00+00:00',
                        '2021-12-13 08:00:00+00:00', '2021-12-14 08:00:00+00:00',
                        '2021-12-23 08:00:00+00:00'],
                       dtype='datetime64[ns, UTC]',
                       name='datetime',
                       freq=None))
    ])
    def test_station_outlier_date(self, station_id, start, stop, source, out_date):
        outlier_storms = OutlierEvents.from_station(
            station_id=station_id,
            start=start, stop=stop, source=source
        )

        outlier_storms.find()
        results = outlier_storms.dates
        assert np.all(results == out_date)

    @pytest.mark.parametrize('station_id, start, stop, source', [
        ('TUM', datetime(2021, 10, 1), datetime(2021, 10, 12), 'CDEC')
    ])
    def test_length_station(self, station_id, start, stop, source):
        outlier_storms = OutlierEvents.from_station(
            station_id=station_id,
            start=start, stop=stop, source=source
        )
        with pytest.raises(ValueError,
                           match='Data length must be greater than 15 '
                                 'for outlier calculation.'):
            outlier_storms.find()
