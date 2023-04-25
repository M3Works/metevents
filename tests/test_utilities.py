import pandas as pd
from datetime import datetime, timedelta
import pytest

from metevents.utilities import determine_freq


@pytest.mark.parametrize('date_data, expected', [
    # monotonic Days
    ([datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)], 'D'),
    # monotonic hours
    ([datetime(2023, 1, 1) + timedelta(hours=i) for i in range(10)], 'H'),
    # Irregular interval
    ([datetime(2023, 1, 1) + timedelta(days=i ** 2) for i in range(10)], None)
])
def test_determine_freq(date_data, expected):
    series = pd.Series(range(len(date_data)), index=date_data)
    freq_str = determine_freq(series)
    assert freq_str == expected
