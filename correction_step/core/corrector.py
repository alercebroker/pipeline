from typing import Literal

import numpy as np
import pandas as pd

from . import strategy


class Corrector:
    """Class for applying corrections to a list of detections"""

    # _EXTRA_FIELDS must include columns from all surveys that are needed in their respective strategy
    _EXTRA_FIELDS = ["magnr", "sigmagnr", "distnr", "distpsnr1", "sgscore1", "sharpnr", "chinr"]
    _ZERO_MAG = 100.0  # Not really zero mag, but zero flux (very high magnitude)

    def __init__(self, detections: list[dict]):
        """Duplicate detections are dropped from all calculations and outputs."""
        self._detections = pd.DataFrame.from_records(detections, exclude={"extra_fields"})
        self._detections = self._detections.drop_duplicates("candid").set_index("candid")

        self.__extras = {alert["candid"]: alert["extra_fields"] for alert in detections}
        extras = pd.DataFrame.from_dict(self.__extras, orient="index", columns=self._EXTRA_FIELDS)
        extras = extras.reset_index(names=["candid"]).drop_duplicates("candid").set_index("candid")

        self._detections = self._detections.join(extras)

    def _survey_mask(self, survey: str):
        """Creates boolean mask of detections whose `tid` starts with given survey name (case-insensitive)

        Args:
            survey: Name of the survey of interest

        Returns:
            pd.Series: Mask of detections corresponding to the survey
        """
        return self._detections["tid"].str.lower().str.startswith(survey.lower())

    def _apply_all_surveys(self, function: str, *, default=None, columns=None, dtype=object):
        """Applies given function for all surveys defined in `strategy` module.

        Any survey without defined strategies will keep the default value.

        Args:
            function: Name of the function to apply. It must exist as a function in all strategy modules
            default (any): Default value for surveys without a defined strategy
            columns (list[str]): Create a dataframe with these columns. If not provided, it will create a series
            dtype (dtype): Type of the output values

        Returns:
            pd.Series or pd.DataFrame: Result of the applied function for all detections
        """
        if columns:
            basic = pd.DataFrame(default, index=self._detections.index, columns=columns, dtype=dtype)
        else:
            basic = pd.Series(default, index=self._detections.index, dtype=dtype)
        for name in strategy.__dict__:  # Will loop through the modules inside strategy
            if name.startswith("_"):  # Skip protected/private modules/variables
                continue
            mask = self._survey_mask(name)  # Module must match survey prefix uniquely
            if mask.any():  # Skip if there are no detections of the given survey
                module = getattr(strategy, name)  # Get module containing strategy for survey
                # Get function and call it over the detections belonging to the survey
                basic[mask] = getattr(module, function)(self._detections[mask])
        return basic.astype(dtype)  # Ensure correct output type

    @property
    def corrected(self) -> pd.Series:
        """Whether the detection has a corrected magnitude"""
        return self._apply_all_surveys("is_corrected", default=False, dtype=bool)

    @property
    def dubious(self) -> pd.Series:
        """Whether the correction (or lack thereof) is dubious"""
        return self._apply_all_surveys("is_dubious", default=False, dtype=bool)

    @property
    def stellar(self) -> pd.Series:
        """Whether the source is likely stellar"""
        return self._apply_all_surveys("is_stellar", default=False, dtype=bool)

    def _correct(self) -> pd.DataFrame:
        """Calculate corrected magnitudes and corrected errors for detections"""
        cols = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
        return self._apply_all_surveys("correct", columns=cols, dtype=float)

    def corrected_dataframe(self) -> pd.DataFrame:
        """Detection dataframe including corrected magnitudes. Only includes generic fields as columns.

        Corrected magnitudes and errors for sources without corrections are set to `None`.
        """
        corrected = self._correct().replace(np.inf, self._ZERO_MAG)
        corrected[~self.corrected] = np.nan
        corrected = corrected.assign(corrected=self.corrected, dubious=self.dubious, stellar=self.stellar)
        return self._detections.join(corrected).replace(np.nan, None).drop(columns=self._EXTRA_FIELDS)

    def corrected_records(self) -> list[dict]:
        """Corrected alerts as records.

        The output is the same as the input passed on creation, with additional generic fields corresponding to
        the corrections (`mag_corr`, `e_mag_corr`, `e_mag_corr_ext`, `corrected`, `dubious`, `stellar`).
        """
        corrected = self.corrected_dataframe().reset_index().to_dict("records")
        return [{**record, "extra_fields": self.__extras[record["candid"]]} for record in corrected]

    @staticmethod
    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        return np.average(values, weights=weights)

    @staticmethod
    def weighted_error(weights: pd.Series) -> float:
        return np.sqrt(1 / np.sum(weights))

    @staticmethod
    def arcsec2dec(values: pd.Series | float) -> pd.Series | float:
        return values / 3600.0

    @staticmethod
    def dec2arcsec(values: pd.Series | float) -> pd.Series | float:
        return values * 3600.0

    def _calculate_coordinates(self, label: Literal["ra", "dec"]) -> dict:
        def _average(series):
            return self.weighted_mean(series, weights.loc[series.index])

        weights = 1 / self.arcsec2dec(self._detections[f"e_{label}"]) ** 2
        return {f"mean{label}": self._detections.groupby("aid")[label].agg(_average)}

    def coordinates_dataframe(self) -> pd.DataFrame:
        coords = self._calculate_coordinates("ra")
        coords.update(self._calculate_coordinates("dec"))
        return pd.DataFrame(coords)

    def coordinates_records(self) -> dict:
        return self.coordinates_dataframe().to_dict("index")
