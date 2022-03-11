from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import Union


@dataclass(frozen=True)
class HistrogramBin:
    start: Union[float, int]
    end: Union[float, int]
    freq: int
    nfreq: float


@dataclass
class EmpiricalDistribution:
    """
    Representation of an empiricial distribution which has no guarantees
    of accuracy, as it's limited by regular floating point precision and
    arithmetic.
    """

    _content: list[Union[float, int]] = field(default_factory=list)
    _min: Union[float, int] = field(init=False)
    _max: Union[float, int] = field(init=False)
    _median: Union[float, int] = field(init=False)
    _mean: Union[float, int] = field(init=False)
    _std: Union[float, int] = field(init=False)

    def __post_init__(self) -> None:
        self._content.sort()
        self._refresh_summary()

    @contextmanager
    def open(self) -> list[Union[float, int]]:
        """
        Function used to modify _content. The context ensures integrity
        of the summary statistics whenever it closes.
        """
        try:
            yield self._content
        finally:
            self._content.sort()
            self._refresh_summary()

    def _refresh_summary(self) -> None:
        if len(self.content) > 0:
            self._min = min(self._content)
            self._max = max(self._content)
            self._median = median(self._content)
            self._mean = mean(self._content)
            self._std = stdev(self._content)

    def bins(self, n: int) -> list[HistrogramBin]:
        bins = []
        interval_size = (self.max - self.min) / n

        start = self.min
        end = interval_size
        freq = 0

        for sample in self.content:
            if sample < end:
                freq += 1
            else:
                bins.append(
                    HistrogramBin(
                        start=start,
                        end=end,
                        freq=freq,
                        nfreq=freq / len(self.content),
                    )
                )
                start = end
                end += interval_size
                freq = 0

        return bins

    @property
    def content(self) -> list[Union[float, int]]:
        return self._content

    @property
    def min(self) -> Union[float, int]:
        return self._min

    @property
    def max(self) -> Union[float, int]:
        return self._max

    @property
    def median(self) -> Union[float, int]:
        return self._median

    @property
    def mean(self) -> Union[float, int]:
        return self._mean

    @property
    def std(self) -> Union[float, int]:
        return self._std
