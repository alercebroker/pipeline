import abc

import numpy as np
import pandas as pd


class BaseStrategy(abc.ABC):
    """The Strategy interface declares operations common to all
    supported versions of some algorithm.

    The Context uses this interface to call the algorithm defined
    by Concrete Strategies.
    """
    ZERO_MAG = 100.  # Not really zero mag, but zero flux (very high magnitude)
    EXTRA_FIELDS = []

    def __init__(self, alerts: list[dict]):
        candid, extras = "candid", "extra_fields"
        self.__extras = {alert[candid]: alert[extras] for alert in alerts}
        self._generic = pd.DataFrame.from_records(alerts, exclude={extras}, index=candid)
        self._extras = pd.DataFrame(self.__extras.values(), index=self.__extras.keys(), columns=self.EXTRA_FIELDS)
        self._extras.index.name = candid

    @property
    @abc.abstractmethod
    def near_source(self) -> pd.Series:
        return pd.Series(False, index=self._generic.index)

    @property
    @abc.abstractmethod
    def dubious(self) -> pd.Series:
        return pd.Series(False, index=self._generic.index)

    @abc.abstractmethod
    def _correct(self) -> pd.DataFrame:
        columns = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
        return pd.DataFrame(np.inf, columns=columns, index=self._generic.index)

    def corrected_frame(self) -> pd.DataFrame:
        corrected = self._correct().replace(np.inf, self.ZERO_MAG)
        corrected[~self.near_source] = np.nan
        alerts = self._generic.join(corrected, lsuffix="_old").replace(np.nan, None)
        return alerts.assign(corrected=self.near_source, dubious=self.dubious)

    def corrected_message(self) -> list[dict]:
        alerts = self.corrected_frame().reset_index().to_dict("records")
        return [{**alert, "extra_fields": self.__extras[alert["candid"]]} for alert in alerts]
