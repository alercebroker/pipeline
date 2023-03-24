import abc

import numpy as np
import pandas as pd


class BaseStrategy(abc.ABC):
    """Define the base strategy for correction

    The abstract methods should be overwritten based on the characteristics of the survey, but a default
    routine is already present. This sets the properties `near_source` and `dubious` to all `False`. The
    corrected dataframe will have only NaN values.

    The class attribute `EXTRA_FIELDS` should be implemented in each subclass. This contains a list of fields
    that are present in the `extra_fields` of the alert and that are needed to compute the correction.
    """
    EXTRA_FIELDS = []
    _ZERO_MAG = 100.  # Not really zero mag, but zero flux (very high magnitude)

    def __init__(self, alerts: list[dict]):
        candid, extras = "candid", "extra_fields"
        self.__extras = {alert[candid]: alert[extras] for alert in alerts}
        self._generic = pd.DataFrame.from_records(alerts, exclude={extras}, index=candid)
        self._extras = pd.DataFrame(self.__extras.values(), index=self.__extras.keys(), columns=self.EXTRA_FIELDS)
        self._extras.index.name = candid

    @property
    @abc.abstractmethod
    def corrected(self) -> pd.Series:
        """Whether the detection has a nearby known source"""
        return pd.Series(False, index=self._generic.index)

    @property
    @abc.abstractmethod
    def dubious(self) -> pd.Series:
        """Whether the correction or lack thereof is dubious"""
        return pd.Series(False, index=self._generic.index)

    @property
    @abc.abstractmethod
    def stellar(self) -> pd.Series:
        """Whether the source is likely stellar"""
        return pd.Series(False, index=self._generic.index)

    @abc.abstractmethod
    def _correct(self) -> pd.DataFrame:
        """Convert difference magnitude into apparent magnitude"""
        columns = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
        return pd.DataFrame(columns=columns, index=self._generic.index)

    def corrected_frame(self, inf_to=_ZERO_MAG) -> pd.DataFrame:
        """Alert dataframe including corrected magnitudes.

        Corrected magnitudes for objects considered far from a nearby source are set to NaN.
        """
        corrected = self._correct().replace(np.inf, inf_to)
        corrected[~self.corrected] = np.nan
        alerts = self._generic.join(corrected, lsuffix="_old").replace(np.nan, None)
        return alerts.assign(corrected=self.corrected, dubious=self.dubious, stellar=self.stellar)

    def corrected_message(self) -> list[dict]:
        """Recreates initialization message, including magnitude corrections"""
        alerts = self.corrected_frame().reset_index().to_dict("records")
        return [{**alert, "extra_fields": self.__extras[alert["candid"]]} for alert in alerts]
