from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from .strategy import is_corrected, is_dubious, is_stellar, correct


class Corrector:
    """Class for applying corrections to a list of detections"""

    # _EXTRA_FIELDS must include columns from all surveys that are needed
    # in their respective strategy

    # new dets will be corrected by Corrector
    _EXTRA_FIELDS_NEW_DET = [
        "magnr",
        "sigmagnr",
        "distnr",
        "distpsnr1",
        "sgscore1",
        "sharpnr",
        "chinr",
    ]
    # old dets are already corrected
    _EXTRA_FIELDS_OLD_DET = ["mag_corr", "e_mag_corr", "e_mag_corr_ext", "corrected", "dubious"]
    # Not really zero mag, but zero flux (very high magnitude)
    _ZERO_MAG = 100.0

    def __init__(self, detections):
        """Creates object that handles detection corrections.

        Duplicate `measurement_ids` are dropped from all calculations and outputs.

        Args:
            detections: List of mappings with all values
                from generic alert (must include `extra_fields`)
        """
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        
        self._detections = detections.drop_duplicates(subset=["measurement_id", "oid"])
        index_key = self._detections["measurement_id"].astype(str) + "_" + self._detections["oid"].astype(str)
        self._detections = self._detections.set_index(index_key)

        self.__extras = self._extract_extra_fields(detections)
        extras = self._create_extras_dataframe()

        
        self._detections = self._detections.join(extras, how="left")

    def _extract_extra_fields(self, detections) -> list[dict]:
        """Extract extra fields from detections DataFrame."""
        extras = []
        for _, row in detections.iterrows():
            extra_dict = {
                "measurement_id": row["measurement_id"],
                "oid": row["oid"],
            }
            
            if isinstance(row["extra_fields"], dict):
                extra_dict.update(row["extra_fields"])
            
            extras.append(extra_dict)
        
        return extras

    def _create_extras_dataframe(self) -> pd.DataFrame:
        """Create and process extras DataFrame."""
        all_columns = (
            self._EXTRA_FIELDS_NEW_DET + 
            self._EXTRA_FIELDS_OLD_DET + 
            ["measurement_id", "oid"]
        )
        
        extras = pd.DataFrame(self.__extras, columns=all_columns)
        extras = extras.drop_duplicates(["measurement_id", "oid"])
        
        index_key = extras["measurement_id"].astype(str) + "_" + extras["oid"].astype(str)
        extras = extras.set_index(index_key)
        extras = extras.drop(columns=["oid", "measurement_id"], errors='ignore')
        
        return extras

    @property
    def corrected(self) -> pd.Series:
        """Whether the detection has a corrected magnitude"""
        return is_corrected(self._detections)

    @property
    def dubious(self) -> pd.Series:
        """Whether the correction (or lack thereof) is dubious"""
        return is_dubious(self._detections)

    @property
    def stellar(self) -> pd.Series:
        """Whether the source is likely stellar"""
        return is_stellar(self._detections)

    def corrected_magnitudes(self) -> pd.DataFrame:
        """Dataframe with corrected magnitudes and errors.
        Non-corrected magnitudes are set to NaN."""
        corrected_df = correct(self._detections)
        # NaN for non-corrected magnitudes
        return corrected_df.where(self.corrected)

    def corrected_as_dataframe(self) -> pd.DataFrame:
        """Corrected alerts as DataFrame.

        The output is the same as the input passed on creation,
        with additional generic fields corresponding to the corrections
        (`mag_corr`, `e_mag_corr`, `e_mag_corr_ext`, `corrected`, `dubious`, `stellar`).

        Returns DataFrame with all the corrected data and proper types preserved.
        """
        self.logger.debug("Correcting %s detections...", len(self._detections))
        
        corrected = self.corrected_magnitudes().replace(np.inf, self._ZERO_MAG)
        corrected = corrected.assign(
            corrected=self.corrected, 
            dubious=self.dubious, 
            stellar=self.stellar
        )

        detections = self._detections.drop(columns=self._EXTRA_FIELDS_OLD_DET, errors='ignore')
        corrected = detections.join(corrected)        

        # Clean NaN values but preserve DataFrame structure
        corrected = (corrected
                    .replace([np.nan, "nan", -np.inf], None)
                    .drop(columns=self._EXTRA_FIELDS_NEW_DET, errors='ignore'))
        
        self.logger.debug("Corrected %s", corrected["corrected"].sum())

        # Reorder columns to match schema
        schema_output_order = [
            "oid", "sid", "pid", "tid", "band", "measurement_id", "mjd",
            "ra", "e_ra", "dec", "e_dec", "mag", "e_mag",
            "mag_corr", "e_mag_corr", "e_mag_corr_ext",
            "isdiffpos", "corrected", "dubious", "stellar",
            "has_stamp", "forced", "new", "parent_candid",
        ]
        
        corrected = corrected[schema_output_order]
        
        # Add extra_fields as a column instead of converting to records
        extras_lookup = {
            (extra["oid"], extra["measurement_id"]): extra 
            for extra in self.__extras
        }
        
        def get_extra_fields(row):
            key = (row["oid"], row["measurement_id"])
            extra_fields = extras_lookup.get(key, {}).copy()
            
            # Remove oid and measurement_id from extra_fields
            extra_fields.pop("measurement_id", None)
            extra_fields.pop("oid", None)
            
            # Clean NaN values in extra_fields
            for k, v in list(extra_fields.items()):
                if isinstance(v, float) and np.isnan(v):
                    extra_fields[k] = None
            
            return extra_fields
        
        # Add extra_fields as a column
        corrected = corrected.reset_index(drop=True)
        corrected["extra_fields"] = corrected.apply(get_extra_fields, axis=1)

        # Return the column measurement id back to its original format after changes necessary for corrector
        corrected["measurement_id"] = corrected["measurement_id"].astype("Int64")
        return corrected

    @staticmethod
    def weighted_mean(values: pd.Series, sigmas: pd.Series) -> float:
        """Compute error weighted mean of values.
        The weights used are the inverse square of the errors.

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
        
        if non_forced.empty:
            return {f"mean{label}": pd.Series(dtype=float)}

        sigmas = self.arcsec2dec(non_forced[f"e_{label}"])
        
        def _average(series):
            return self.weighted_mean(series, sigmas.loc[series.index])

        return {f"mean{label}": non_forced.groupby("oid")[label].agg(_average)}

    def mean_coordinates(self) -> pd.DataFrame:
        """Dataframe with weighted mean coordinates for each OID"""
        coords = self._calculate_coordinates("ra")
        coords.update(self._calculate_coordinates("dec"))
        return pd.DataFrame(coords)

    def coordinates_as_dataframe(self) -> pd.DataFrame:
        """Weighted mean coordinates as DataFrame with OID as a column"""
        coords_df = self.mean_coordinates()
        coords_df = coords_df.reset_index()
        return coords_df



    





















