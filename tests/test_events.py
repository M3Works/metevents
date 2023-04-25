from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest
from pandas import DatetimeIndex

from metevents.events import StormEvents, BaseEvents


@pytest.fixture()
def series(data):
    index = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(len(data))]
    return pd.Series(data, index=DatetimeIndex(index, freq='D'))


@pytest.mark.parametrize('data, expected', [
    ([True, True, False], [timedelta(days=2), timedelta(days=2),
                           np.datetime64('NaT')]),
])
def test_get_start_stop(series, data, expected):
    start_stop = BaseEvents.get_timedelta(series)
    exp_series = pd.Series(expected, index=start_stop.index)
    pd.testing.assert_series_equal(start_stop, exp_series, check_index=False)


@pytest.mark.parametrize('data, mass, hours, n_storms', [
    # Two storms
    ([0, 1, 1, 0, 0, 1, 1], 0.1, 24, 2),
    # Different storm def, emphasizing hours to end, 1 storm
    ([0, 1, 1, 0.1, 0, 1, 0], 0.1, 48, 1),
    # No break in the storm
    ([1, 1, 1], 0.1, 24, 1),
    # Storm split by mass only
    ([1, 0.5, 0.1, 0.2, 1], 0.5, 24, 2)
])
def test_storm_events(series, data, mass, hours, n_storms):
    """
    Test the number of storms identified by varying input data
    and thresholds.
    """
    storms = StormEvents(series)
    storms.find(mass_to_start=mass, hours_to_stop=hours)
    assert storms.N == n_storms


@pytest.mark.parametrize('station_id, start, stop, source, mass, hours, n_storms', [
    ('TUM', datetime(2021, 12, 1), datetime(2022, 1, 15), 'CDEC', 0.1, 48, 3),
    ('637:ID:SNTL', datetime(2022, 12, 1, ), datetime(2022, 12, 15),
     'NRCS', 0.1, 48, 2)

])
def test_storm_events_from_station(station_id, start, stop, source, mass, hours,
                                   n_storms):
    """
    Test the number of storms identified by varying input data and thresholds.
    """
    storms = StormEvents.from_station(station_id, start, stop, source=source)
    storms.find(mass_to_start=mass, hours_to_stop=hours)
    assert storms.N == n_storms
