import datetime
def determine_freq(series):
    """
    If the frequency string is not known, try to figure it out.
    Args:
        series: datetime indexed pandas series.
    Returns:
        freq: time frequency string, None if it is unknown.
    """
    freq = series.index.freqstr
    if freq is None:
        result = (series.index[1:-1] - series.index[0:-2]).unique()
        if len(result) == 1:
            freq = result[0].resolution_string
    return freq

def current_water_year():
    today = datetime.date.today()
    if today.month >= 10:
        return today.year + 1
    else:
        return today.year