from .base import GenericPreprocessor
import numpy as np
import pandas as pd


def probs_to_normal_form(probs):
    list_of_df = []
    for alerce_class in probs.columns:
        class_probs = probs[[alerce_class]].copy()
        class_probs.rename(columns={alerce_class: "probability"}, inplace=True)
        class_probs["classALeRCE"] = alerce_class
        list_of_df.append(class_probs)
    final_table = pd.concat(list_of_df, axis=0)
    return final_table


# This function mimics the behavior of the elasticc stream, not LSST's
def short_lightcurve_mask(times, photflags, n_days):
    first_mjd = times[(photflags & 2048) != 0][0]
    time_limit = first_mjd + n_days
    valid_time_mask = (times < time_limit) & (times > first_mjd - 30)

    detected_mask = (photflags & 4096) != 0
    last_detection_time = times[detected_mask & valid_time_mask][-1]
    before_last_detection_mask = times <= last_detection_time

    final_mask = (detected_mask & valid_time_mask) | (
        (~detected_mask) & before_last_detection_mask & valid_time_mask
    )
    return final_mask, time_limit


def shorten_lightcurve(df, n_days):
    mask, _ = short_lightcurve_mask(df["MJD"].values, df["PHOTFLAG"].values, n_days)
    return df[mask]


class ElasticcPreprocessor(GenericPreprocessor):
    """Preprocessing for lightcurves from ZTF forced photometry service."""

    def __init__(self, stream=False):
        super().__init__()
        self.stream = stream

        self.required_columns = ["MJD", "FLUXCAL", "FLUXCALERR", "BAND"]

        self.column_translation = {
            "MJD": "time",
            "BAND": "band",
            "FLUXCAL": "difference_flux",
            "FLUXCALERR": "difference_flux_error",
        }

        self.metadata_column_map = {
            "hostgal2_ellipticity": "HOSTGAL2_ELLIPTICITY",
            "hostgal2_mag_Y": "HOSTGAL2_MAG_Y",
            "hostgal2_mag_g": "HOSTGAL2_MAG_g",
            "hostgal2_mag_i": "HOSTGAL2_MAG_i",
            "hostgal2_mag_r": "HOSTGAL2_MAG_r",
            "hostgal2_mag_u": "HOSTGAL2_MAG_u",
            "hostgal2_mag_z": "HOSTGAL2_MAG_z",
            "hostgal2_magerr_Y": "HOSTGAL2_MAGERR_Y",
            "hostgal2_magerr_g": "HOSTGAL2_MAGERR_g",
            "hostgal2_magerr_i": "HOSTGAL2_MAGERR_i",
            "hostgal2_magerr_r": "HOSTGAL2_MAGERR_r",
            "hostgal2_magerr_u": "HOSTGAL2_MAGERR_u",
            "hostgal2_magerr_z": "HOSTGAL2_MAGERR_z",
            "hostgal2_snsep": "HOSTGAL2_SNSEP",
            "hostgal2_sqradius": "HOSTGAL2_SQRADIUS",
            "hostgal2_zphot": "HOSTGAL2_PHOTOZ",
            "hostgal2_zphot_err": "HOSTGAL2_PHOTOZ_ERR",
            "hostgal2_zphot_q000": "HOSTGAL2_ZPHOT_Q000",
            "hostgal2_zphot_q010": "HOSTGAL2_ZPHOT_Q010",
            "hostgal2_zphot_q020": "HOSTGAL2_ZPHOT_Q020",
            "hostgal2_zphot_q030": "HOSTGAL2_ZPHOT_Q030",
            "hostgal2_zphot_q040": "HOSTGAL2_ZPHOT_Q040",
            "hostgal2_zphot_q050": "HOSTGAL2_ZPHOT_Q050",
            "hostgal2_zphot_q060": "HOSTGAL2_ZPHOT_Q060",
            "hostgal2_zphot_q070": "HOSTGAL2_ZPHOT_Q070",
            "hostgal2_zphot_q080": "HOSTGAL2_ZPHOT_Q080",
            "hostgal2_zphot_q090": "HOSTGAL2_ZPHOT_Q090",
            "hostgal2_zphot_q100": "HOSTGAL2_ZPHOT_Q100",
            "hostgal2_zspec": "HOSTGAL2_SPECZ",
            "hostgal2_zspec_err": "HOSTGAL2_SPECZ_ERR",
            "hostgal_ellipticity": "HOSTGAL_ELLIPTICITY",
            "hostgal_mag_Y": "HOSTGAL_MAG_Y",
            "hostgal_mag_g": "HOSTGAL_MAG_g",
            "hostgal_mag_i": "HOSTGAL_MAG_i",
            "hostgal_mag_r": "HOSTGAL_MAG_r",
            "hostgal_mag_u": "HOSTGAL_MAG_u",
            "hostgal_mag_z": "HOSTGAL_MAG_z",
            "hostgal_magerr_Y": "HOSTGAL_MAGERR_Y",
            "hostgal_magerr_g": "HOSTGAL_MAGERR_g",
            "hostgal_magerr_i": "HOSTGAL_MAGERR_i",
            "hostgal_magerr_r": "HOSTGAL_MAGERR_r",
            "hostgal_magerr_u": "HOSTGAL_MAGERR_u",
            "hostgal_magerr_z": "HOSTGAL_MAGERR_z",
            "hostgal_snsep": "HOSTGAL_SNSEP",
            "hostgal_sqradius": "HOSTGAL_SQRADIUS",
            "hostgal_zphot": "HOSTGAL_PHOTOZ",
            "hostgal_zphot_err": "HOSTGAL_PHOTOZ_ERR",
            "hostgal_zphot_q000": "HOSTGAL_ZPHOT_Q000",
            "hostgal_zphot_q010": "HOSTGAL_ZPHOT_Q010",
            "hostgal_zphot_q020": "HOSTGAL_ZPHOT_Q020",
            "hostgal_zphot_q030": "HOSTGAL_ZPHOT_Q030",
            "hostgal_zphot_q040": "HOSTGAL_ZPHOT_Q040",
            "hostgal_zphot_q050": "HOSTGAL_ZPHOT_Q050",
            "hostgal_zphot_q060": "HOSTGAL_ZPHOT_Q060",
            "hostgal_zphot_q070": "HOSTGAL_ZPHOT_Q070",
            "hostgal_zphot_q080": "HOSTGAL_ZPHOT_Q080",
            "hostgal_zphot_q090": "HOSTGAL_ZPHOT_Q090",
            "hostgal_zphot_q100": "HOSTGAL_ZPHOT_Q100",
            "hostgal_zspec": "HOSTGAL_SPECZ",
            "hostgal_zspec_err": "HOSTGAL_SPECZ_ERR",
            "mwebv": "MWEBV",
            "mwebv_err": "MWEBV_ERR",
            "z_final": "REDSHIFT_HELIO",
            "z_final_err": "REDSHIFT_HELIO_ERR",
        }

        self.new_columns = []
        for c in self.required_columns:
            if c in self.column_translation.keys():
                self.new_columns.append(self.column_translation[c])
            else:
                self.new_columns.append(c)

    def has_necessary_columns(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        input_columns = set(dataframe.columns)
        constraint = set(self.required_columns)
        difference = constraint.difference(input_columns)
        return len(difference) == 0

    def discard_noisy_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections[
            (
                (detections["difference_flux_error"] < 300)
                & (np.abs(detections["difference_flux"]) < 50e3)
            )
        ]
        return detections

    def preprocess(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        self.verify_dataframe(dataframe)
        if not self.has_necessary_columns(dataframe):
            raise Exception(
                "Lightcurve dataframe does not have all the necessary columns"
            )

        dataframe = self.rename_columns_detections(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)

        # A little hack ;)
        dataframe["magnitude"] = dataframe["difference_flux"]
        dataframe["error"] = dataframe["difference_flux_error"]

        return dataframe

    def preprocess_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        if not self.stream:
            return metadata

        return metadata.rename(columns=self.metadata_column_map)

    def rename_columns_detections(self, detections: pd.DataFrame):
        return detections.rename(
            columns=self.column_translation, errors="ignore", inplace=False
        )
