from __future__ import annotations

import logging
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
        """Creates objet that handles detection corrections.

        Duplicate `candids` are dropped from all calculations and outputs.

        Args:
            detections: List of mappings with all values from generic alert (must include `extra_fields`)
        """
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        self._detections = pd.DataFrame.from_records(detections, exclude={"extra_fields"})
        self._detections = self._detections.drop_duplicates("candid").set_index("candid")

        self.__extras = {alert["candid"]: alert["extra_fields"] for alert in detections}
        extras = pd.DataFrame.from_dict(self.__extras, orient="index", columns=self._EXTRA_FIELDS)
        extras = extras.reset_index(names=["candid"]).drop_duplicates("candid").set_index("candid")

        self._detections = self._detections.join(extras)

    def _survey_mask(self, survey: str):
        """Creates boolean mask of detections whose `sid` matches the given survey name (case-insensitive)

        Args:
            survey: Name of the survey of interest

        Returns:
            pd.Series: Mask of detections corresponding to the survey
        """
        return self._detections["sid"].str.lower() == survey.lower()

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
        for name in dir(strategy):  # Will loop through the modules/variables imported in strategy
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

    def corrected_magnitudes(self) -> pd.DataFrame:
        """Dataframe with corrected magnitudes and errors. Non-corrected magnitudes are set to NaN."""
        cols = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
        corrected = self._apply_all_surveys("correct", columns=cols, dtype=float)
        return corrected.where(self.corrected)  # NaN for non-corrected magnitudes

    def corrected_as_records(self) -> list[dict]:
        """Corrected alerts as records.

        The output is the same as the input passed on creation, with additional generic fields corresponding to
        the corrections (`mag_corr`, `e_mag_corr`, `e_mag_corr_ext`, `corrected`, `dubious`, `stellar`).

        The records are a list of mappings with the original input pairs and the new pairs together.
        """
        self.logger.debug(f"Correcting {len(self._detections)} detections...")
        corrected = self.corrected_magnitudes().replace(np.inf, self._ZERO_MAG)
        corrected = corrected.assign(corrected=self.corrected, dubious=self.dubious, stellar=self.stellar)
        corrected = self._detections.join(corrected).replace(np.nan, None).drop(columns=self._EXTRA_FIELDS)
        corrected = corrected.replace(-np.inf, None)
        self.logger.debug(f"Corrected {corrected['corrected'].sum()}")
        corrected = corrected.reset_index().to_dict("records")
        return [{**record, "extra_fields": self.__extras[record["candid"]]} for record in corrected]

    @staticmethod
    def weighted_mean(values: pd.Series, sigmas: pd.Series) -> float:
        """Compute error weighted mean of values. The weights used are the inverse square of the errors.

        Args:
            values: Values for which to compute the mean
            sigmas: Errors associated with the values

        Returns:
            float: Weighted mean of the values
        """
        return np.average(values, weights=1 / sigmas**2)

    @staticmethod
    def arcsec2dec(values: pd.Series | float) -> pd.Series | float:
        """Converts values from arc-seconds to degrees.

        Args:
            values: Value in arcsec

        Returns:
            pd.Series | float: Value in degrees
        """
        return values / 3600.0

    def _calculate_coordinates(self, label: Literal["ra", "dec"]) -> dict:
        """Calculate weighted mean value for the given coordinates for each OID.

        Args:
            label: Label for the coordinate to calculate

        Returns:
            dict: Mapping from `mean<label>` to weighted means of the coordinates
        """
        non_forced = self._detections[~self._detections["forced"]]

        def _average(series):
            return self.weighted_mean(series, sigmas.loc[series.index])

        sigmas = self.arcsec2dec(non_forced[f"e_{label}"])
        return {f"mean{label}": non_forced.groupby("oid")[label].agg(_average)}

    def mean_coordinates(self) -> pd.DataFrame:
        """Dataframe with weighted mean coordinates for each OID"""
        coords = self._calculate_coordinates("ra")
        coords.update(self._calculate_coordinates("dec"))
        return pd.DataFrame(coords)

    def coordinates_as_records(self) -> dict:
        """Weighted mean coordinates as records (mapping from OID to a mapping of mean coordinates)"""
        return self.mean_coordinates().to_dict("index")
