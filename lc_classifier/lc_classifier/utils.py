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
