class BaseTimePeriod:
    """
    Class for holding on to periods of timeseries data
    that meets criteria
    """
    def __init__(self, data):
        self._data = data
        self._start = None
        self._stop = None
        self._duration = None

    @property
    def start(self):
        if self._start is None:
            self._start = self.data.index.min()
        return self._start

    @property
    def stop(self):
        if self._stop is None:
            self._stop = self.data.index.max()
        return self._stop

    @property
    def duration(self):
        if self._duration is None:
            self._duration = self.stop - self.start
        return self._duration

    @property
    def data(self):
        return self._data


class CumulativePeriod(BaseTimePeriod):
    def __init__(self, data):
        super().__init__(data)
        self._total = None

    @property
    def total(self):
        if self._total is None:
            self._total = self._data.sum()

        return self._total

    def __repr__(self):
        return f"Cumulative Period ({self.start.isoformat()} - {self.stop.isoformat()})"
