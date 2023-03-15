import abc
import pandas as pd


class BaseStrategy(abc.ABC):
    """The Strategy interface declares operations common to all
    supported versions of some algorithm.

    The Context uses this interface to call the algorithm defined
    by Concrete Strategies.
    """
    EXTRA_FIELDS = []

    def __init__(self, alerts: list[dict]):
        extra_fields = [alert["extra_fields"] for alert in alerts]
        self._generic = pd.DataFrame.from_records(alerts, exclude={"extra_fields"}, index="candid")
        self._extra = pd.DataFrame(extra_fields, index=self._generic.index, columns=self.EXTRA_FIELDS)

    @abc.abstractmethod
    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
