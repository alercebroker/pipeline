import abc

import numpy as np
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

    @property
    @abc.abstractmethod
    def corrected(self) -> pd.Series:
        return pd.Series(False, index=self._generic.index)

    @property
    @abc.abstractmethod
    def dubious(self) -> pd.Series:
        return pd.Series(False, index=self._generic.index)

    @abc.abstractmethod
    def _correct(self) -> pd.DataFrame:
        columns = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
        return pd.DataFrame(columns=columns, index=self._generic.index)

    def do_correction(self) -> list[dict]:
        full = self._generic.join(self._correct()).replace(np.nan, None)
        return full.assign(corrected=self.corrected, dubious=self.dubious).to_dict("records")
