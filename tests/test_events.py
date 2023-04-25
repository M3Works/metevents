from datetime import datetime, timedelta

import pandas as pd
import pytest
from pandas import DatetimeIndex, Timestamp

from metevents.events import StormEvents, BaseEvents


@pytest.fixture()
def series(data):
    index = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(len(data))]
    return pd.Series(data, index=DatetimeIndex(index, freq='D'))


@pytest.mark.parametrize('data, expected', [
    ([False, True, True, False], [[Timestamp(2023, 1, 2), Timestamp(2023, 1, 3)]]),
])
def test_get_start_stop(series, data, expected):
    start_stop = BaseEvents.get_start_stop(series)
    exp_series = pd.Series(expected)
    pd.testing.assert_series_equal(start_stop, exp_series, check_index=False)


@pytest.mark.parametrize('data, expected', [
    # Two storms
    ([0, 1, 1, 0, 0, 1, 1], 2)
])
def test_storm_events(series, data, expected):
    strms = StormEvents.from_series(series)
    strms.find()
    assert strms.N == expected
