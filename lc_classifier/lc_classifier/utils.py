from typing import List

import numba
import numpy as np
import pandas as pd

from lc_classifier.features.core.base import AstroObject
import matplotlib.pyplot as plt


@numba.jit(nopython=True)
def is_sorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def plot_astro_object(
    astro_object: AstroObject, unit: str, use_forced_phot: bool, period=None
):
    detections = astro_object.detections
    detections = detections[detections["unit"] == unit]

    available_bands = np.unique(detections["fid"])
    available_bands = set(available_bands)

    if use_forced_phot:
        forced_phot = astro_object.forced_photometry
        forced_phot = forced_phot[forced_phot["unit"] == unit]

        forced_phot_bands = np.unique(forced_phot["fid"])
        forced_phot_bands = set(forced_phot_bands)

        available_bands = available_bands.union(forced_phot_bands)

    color_map = {
        "u": "blue",
        "g": "tab:green",
        "r": "tab:red",
        "i": "tab:purple",
        "z": "tab:brown",
        "Y": "black",
    }

    available_bands = [b for b in list("ugrizY") if b in available_bands]
    for band in available_bands:
        band_detections = detections[detections["fid"] == band]
        band_time = band_detections["mjd"]
        if period is not None:
            band_time = band_time % period

        plt.errorbar(
            band_time,
            band_detections["brightness"],
            yerr=band_detections["e_brightness"],
            fmt="*",
            label=band,
            color=color_map[band],
        )

        if use_forced_phot:
            band_forced = forced_phot[forced_phot["fid"] == band]
            band_forced_time = band_forced["mjd"]
            if period is not None:
                band_forced_time = band_forced_time % period

            plt.errorbar(
                band_forced_time,
                band_forced["brightness"],
                yerr=band_forced["e_brightness"],
                fmt=".",
                label=band + " forced",
                color=color_map[band],
            )

    aid = astro_object.metadata[astro_object.metadata["name"] == "aid"]["value"].values[
        0
    ]
    plt.title(aid)
    plt.xlabel("Time [mjd]")
    plt.ylabel(f"Brightness [{unit}]")
    if unit == "magnitude":
        plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


def all_features_from_astro_objects(astro_objects: List[AstroObject]) -> pd.DataFrame:
    first_object = astro_objects[0]
    features = first_object.features.drop_duplicates(subset=["name", "fid"])
    features = features.set_index(["name", "fid"])
    indexes = features.index.values

    feature_list = []
    aids = []
    for astro_object in astro_objects:
        features = astro_object.features.drop_duplicates(subset=["name", "fid"])
        features = features.set_index(["name", "fid"])
        feature_list.append(features.loc[indexes]["value"].values)

        metadata = astro_object.metadata
        aid = metadata[metadata["name"] == "aid"]["value"].values[0]
        aids.append(aid)

    df = pd.DataFrame(
        data=np.stack(feature_list, axis=0),
        index=aids,
        columns=["_".join([str(i) for i in pair]) for pair in indexes],
    )
    return df


def flux2mag(flux):
    """flux in uJy to AB magnitude"""
    return -2.5 * np.log10(flux) + 23.9


def flux_err_2_mag_err(flux_err, flux):
    return (2.5 * flux_err) / (np.log(10.0) * flux)


def mag2flux(mag):
    return 10 ** (-(mag - 23.9) / 2.5)


def mag_err_2_flux_err(mag_err, mag):
    return np.log(10.0) * mag2flux(mag) / 2.5 * mag_err


def create_astro_object(
    data_origin: str,
    detections: pd.DataFrame,
    forced_photometry: pd.DataFrame,
    xmatch: pd.DataFrame = None,
    non_detections: pd.DataFrame = None,
) -> AstroObject:

    if data_origin == "database":
        mag_corr_column = "magpsf_corr"
        e_mag_corr_ext_column = "sigmapsf_corr_ext"
        diff_mag_column = "magpsf"
        e_diff_mag_column = "sigmapsf"

        mag_corr_column_fp = "mag_corr"
        e_mag_corr_ext_column_fp = "e_mag_corr_ext"
        diff_mag_column_fp = "mag"
        e_diff_mag_column_fp = "e_mag"
    elif data_origin == "explorer":
        mag_corr_column = "mag_corr"
        e_mag_corr_ext_column = "e_mag_corr_ext"
        diff_mag_column = "mag"
        e_diff_mag_column = "e_mag"

        mag_corr_column_fp = "mag_corr"
        e_mag_corr_ext_column_fp = "e_mag_corr_ext"
        diff_mag_column_fp = "mag"
        e_diff_mag_column_fp = "e_mag"

    else:
        raise ValueError(f"{data_origin} is not a valid data_origin")

    detection_keys = [
        "oid",
        "candid",
        "pid",
        "ra",
        "dec",
        "mjd",
        mag_corr_column,
        e_mag_corr_ext_column,
        diff_mag_column,
        e_diff_mag_column,
        "fid",
        "isdiffpos",
    ]

    detections = detections[detection_keys].copy()
    detections["forced"] = False

    forced_photometry["candid"] = forced_photometry["oid"] + forced_photometry[
        "pid"
    ].astype(str)
    forced_photometry_keys = [
        "oid",
        "candid",
        "pid",
        "ra",
        "dec",
        "mjd",
        mag_corr_column_fp,
        e_mag_corr_ext_column_fp,
        diff_mag_column_fp,
        e_diff_mag_column_fp,
        "fid",
        "isdiffpos",
    ]
    forced_photometry = forced_photometry[forced_photometry_keys]
    forced_photometry = forced_photometry[
        (forced_photometry[e_diff_mag_column_fp] != 100)
        | (forced_photometry[diff_mag_column_fp] != 100)
    ].copy()
    forced_photometry["forced"] = True

    # standard names
    detections.rename(
        columns={
            mag_corr_column: "mag_corr",
            e_mag_corr_ext_column: "e_mag_corr_ext",
            diff_mag_column: "mag",
            e_diff_mag_column: "e_mag",
        },
        inplace=True,
    )
    forced_photometry.rename(
        columns={
            mag_corr_column_fp: "mag_corr",
            e_mag_corr_ext_column_fp: "e_mag_corr_ext",
            diff_mag_column_fp: "mag",
            e_diff_mag_column_fp: "e_mag",
        },
        inplace=True,
    )

    a = pd.concat([detections, forced_photometry])
    a["aid"] = "aid_" + a["oid"]
    a["tid"] = "ZTF"
    a["sid"] = "ZTF"
    a.fillna(value=np.nan, inplace=True)
    a.rename(
        columns={"mag_corr": "brightness", "e_mag_corr_ext": "e_brightness"},
        inplace=True,
    )
    a["unit"] = "magnitude"
    a_flux = a.copy()
    a_flux["brightness"] = mag2flux(a["mag"]) * a["isdiffpos"]
    a_flux["e_brightness"] = mag_err_2_flux_err(a["e_mag"], a["mag"])
    a_flux["unit"] = "diff_flux"
    a = pd.concat([a, a_flux], axis=0)
    del a["mag"], a["e_mag"]
    a.set_index("aid", inplace=True)
    a["fid"] = a["fid"].map({1: "g", 2: "r", 3: "i"})
    a = a[a["fid"].isin(["g", "r"])]

    aid = a.index.values[0]
    oid = a["oid"].iloc[0]

    aid_forced = a[a["forced"]]
    aid_detections = a[~a["forced"]]

    if xmatch is None:
        extra_metadata = []
    else:
        extra_metadata = [
            ["W1", xmatch["w1mpro"]],
            ["W2", xmatch["w2mpro"]],
            ["W3", xmatch["w3mpro"]],
            ["W4", xmatch["w4mpro"]],
            ["sgscore1", xmatch["sgscore1"]],
            ["sgmag1", xmatch["sgmag1"]],
            ["srmag1", xmatch["srmag1"]],
            ["distpsnr1", xmatch["distpsnr1"]],
        ]

    metadata = pd.DataFrame(
        [
            ["aid", aid],
            ["oid", oid],
        ]
        + extra_metadata,
        columns=["name", "value"],
    ).fillna(value=np.nan)

    if non_detections is not None:
        non_detections = non_detections[
            [
                "tid",
                "mjd",
                "fid",
                "diffmaglim",
            ]
        ].copy()
        non_detections.rename(columns={"diffmaglim": "brightness"}, inplace=True)

    astro_object = AstroObject(
        detections=aid_detections, forced_photometry=aid_forced, metadata=metadata
    )
    return astro_object
