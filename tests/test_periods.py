import pytest
from datetime import datetime, timedelta
from pandas import Series
import numpy as np

from metevents.periods import BaseTimePeriod, CumulativePeriod


class TestBaseTimePeriod:
    @pytest.fixture()
    def period(self, data):
        index = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(len(data))]
        series = Series(data, index=index)
        return BaseTimePeriod(series)

    @pytest.mark.parametrize('data, expected', [
        ([1, 1, 2, 2], datetime(2023, 1, 1))
    ])
    def test_start(self, period, data, expected):
        period.start == expected

    @pytest.mark.parametrize('data, expected', [
        ([1, 1, 2, 2], datetime(2023, 1, 4))
    ])
    def test_stop(self, period, data, expected):
        period.stop == expected

    @pytest.mark.parametrize('data, expected', [
        ([1, 1, 2, 2], timedelta(days=3))
    ])
    def test_duration(self, period, data, expected):
        assert period.duration == expected


class TestCumulativePeriod:
    @pytest.fixture()
    def period(self, data):
        index = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(len(data))]
        series = Series(data, index=index)
        return CumulativePeriod(series)

    @pytest.mark.parametrize('data, expected', [
        ([1, 1, 2, 2], 6),
        ([1, np.NaN, 2, 2], 5)
    ])
    def test_end(self, period, data, expected):
        assert period.total == expected
