from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from core.strategy import is_corrected, is_dubious, is_stellar, correct


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
        """Creates objet that handles detection corrections.

        Duplicate `measurement_ids` are dropped from all calculations and outputs.

        Args:
            detections: List of mappings with all values
                from generic alert (must include `extra_fields`)
        """
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        self._detections = detections.drop_duplicates(subset=["measurement_id", "oid"])
        self._detections = self._detections.drop_duplicates(["measurement_id", "oid"])
        self._detections = self._detections.set_index(
            self._detections["measurement_id"].astype(str)
            + "_"
            + self._detections["oid"].astype(str)
        )

        import ast

        self.__extras = [
            dict(
                list(row["extra_fields"].items()) if isinstance(row["extra_fields"], dict) else [],
                measurement_id=row["measurement_id"],
                oid=row["oid"],
            )
            for _, row in detections.iterrows()
        ]

        extras = pd.DataFrame(
            self.__extras,
            columns=self._EXTRA_FIELDS_NEW_DET
            + self._EXTRA_FIELDS_OLD_DET
            + ["measurement_id", "oid"],
        )
        extras = extras.drop_duplicates(["measurement_id", "oid"]).set_index(
            self._detections["measurement_id"].astype(str)
            + "_"
            + self._detections["oid"].astype(str)
        )
        self._detections = self._detections.join(extras, how="left", rsuffix="_extra")
        self._detections = self._detections.drop("oid_extra", axis=1)
        self._detections = self._detections.drop("measurement_id_extra", axis=1)

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
    

    def corrected_as_records(self) -> list[dict]:
        """Corrected alerts as records.

        The output is the same as the input passed on creation,
        with additional generic fields corresponding to the corrections
        (`mag_corr`, `e_mag_corr`, `e_mag_corr_ext`, `corrected`, `dubious`, `stellar`).

        The records are a list of mappings with the original input pairs and the new pairs together.
        """

        def find_extra_fields(oid, measurement_id):
            for i, extra in enumerate(self.__extras):
                if extra["oid"] == oid and extra["measurement_id"] == measurement_id:
                    return self.__extras.pop(i)
            return {}
        
        def reorder_record(record):
            # Schema order of fields
            schema_fields = [
                "oid", "sid", "pid", "tid", "band", "measurement_id", "mjd",
                "ra", "e_ra", "dec", "e_dec", "mag", "e_mag", "mag_corr", "e_mag_corr",
                "e_mag_corr_ext", "isdiffpos", "corrected", "dubious", "stellar", "has_stamp",
                "forced", "new", "parent_candid", "extra_fields"
            ]
            
            # Reorder fields of records according to schema
            reordered = {
                "oid": record.get("oid"),
                "sid": record.get("sid"),
                "pid": record.get("pid"),
                "tid": record.get("tid"),
                "band": record.get("band"),
                "measurement_id": [record.get("measurement_id")],  
                "mjd": record.get("mjd"),
                "ra": record.get("ra"),
                "e_ra": record.get("e_ra"),
                "dec": record.get("dec"),
                "e_dec": record.get("e_dec"),
                "mag": record.get("mag"),
                "e_mag": record.get("e_mag"),
                "mag_corr": record.get("mag_corr", None),
                "e_mag_corr": record.get("e_mag_corr", None),
                "e_mag_corr_ext": record.get("e_mag_corr_ext", None),
                "isdiffpos": record.get("isdiffpos"),
                "corrected": record.get("corrected"),
                "dubious": record.get("dubious"),
                "stellar": record.get("stellar"),
                "has_stamp": record.get("has_stamp"),
                "forced": record.get("forced"),
                "new": record.get("new"),
                "parent_candid": record.get("parent_candid", None),
                "extra_fields": record.get("extra_fields", {})
            }
            

            return reordered

        self.logger.debug("Correcting %s detections...", len(self._detections))
        corrected = self.corrected_magnitudes().replace(np.inf, self._ZERO_MAG)
        corrected = corrected.assign(
            corrected=self.corrected, dubious=self.dubious, stellar=self.stellar
        )

        detections = self._detections.drop(columns=self._EXTRA_FIELDS_OLD_DET)
        corrected = detections.join(corrected).replace(np.nan, None).replace("nan", None)
        corrected = corrected.drop(columns=self._EXTRA_FIELDS_NEW_DET)
        corrected = corrected.replace(-np.inf, None)
        self.logger.debug("Corrected %s", corrected["corrected"].sum())
        
        # Reorder the DataFrame columns to match the schema
        desired_order = [
            "oid", "sid", "pid", "tid", "band", "measurement_id", "mjd",
            "ra", "e_ra", "dec", "e_dec", "mag", "e_mag",
            "mag_corr", "e_mag_corr", "e_mag_corr_ext", "isdiffpos",
            "corrected", "dubious", "stellar", "has_stamp", "forced", "new",
            "parent_candid"
        ]
        corrected = corrected[desired_order]

        corrected = corrected.reset_index(drop=True).to_dict("records")
        for record in corrected:
            record["extra_fields"] = find_extra_fields(record["oid"],record["measurement_id"])
            record["extra_fields"].pop("measurement_id", None)
            record["extra_fields"].pop("oid", None)

            for key, value in list(record["extra_fields"].items()):
                if isinstance(value, float) and np.isnan(value):
                    record["extra_fields"][key] = None
                
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
        """Weighted mean coordinates as records
        (mapping from OID to a mapping of mean coordinates)"""
        return self.mean_coordinates().to_dict("index")