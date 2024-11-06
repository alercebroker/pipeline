import os

import numpy as np
import pandas as pd
from lc_classifier.features.core.base import AstroObject, save_astro_objects_batch
from tqdm import tqdm


class NoDetections(Exception):
    pass


def create_astro_object(lc_df: pd.DataFrame, object_info: pd.Series) -> AstroObject:

    lc_df = lc_df.copy()
    # lc_df['detected'] = np.abs(lc_df['flux_diff_ujy'] / lc_df['sigma_flux_diff_ujy']) > 5
    
    lc_df.rename(columns={"dnearestrefsrc": "distnr",
                          "nearestrefsharp": "sharpnr",
                          "nearestrefchi": "chinr"},
                 inplace=True)

    lc_df["fid"] = lc_df["fid"].map({1: "g", 2: "r", 3: "i"})
    lc_df = lc_df[lc_df["fid"].isin(["g", "r"])]

    if len(lc_df[lc_df["detected"]]) == 0:
        raise NoDetections()

    diff_flux = lc_df[
        ["index", "flux_diff_ujy", "sigma_flux_diff_ujy", "fid", "mjd", "detected",
         "distnr", "rfid", "sharpnr", "chinr"]
    ].copy()

    diff_flux.rename(
        columns={"flux_diff_ujy": "brightness", "sigma_flux_diff_ujy": "e_brightness"},
        inplace=True,
    )
    diff_flux["unit"] = "diff_flux"

    magnitude = lc_df[
        ["index", "mag_tot", "sigma_mag_tot", "fid", "mjd", "detected",
         "distnr", "rfid", "sharpnr", "chinr"]
    ].copy()
    magnitude.rename(
        columns={"mag_tot": "brightness", "sigma_mag_tot": "e_brightness"}, inplace=True
    )
    magnitude["unit"] = "magnitude"

    df = pd.concat([diff_flux, magnitude], axis=0)
    df["sid"] = "ztf"
    df["tid"] = "ztf_telescope"
    df.rename(columns={"index": "candid"}, inplace=True)
    df["pid"] = "ztf_forced_phot_service"

    df["ra"] = object_info["ra"]
    df["dec"] = object_info["dec"]

    df.dropna(subset=["brightness", "e_brightness"], inplace=True)

    first_detected_mjd = np.min(df[df["detected"]]["mjd"])
    last_detected_mjd = np.max(df[df["detected"]]["mjd"])

    # These lines are redundant if the input data includes such cuts
    df = df[df["mjd"] <= last_detected_mjd]
    df = df[df["mjd"] >= (first_detected_mjd - 30)]
    
    reference = df[["rfid", "sharpnr", "chinr"]].copy()
    reference.drop_duplicates(keep="first", inplace=True)
    df.drop(columns={"sharpnr", "chinr"},
            inplace=True)

    metadata = pd.DataFrame(
        [
            ["aid", "aid_" + df.index.values[0]],
            ["oid", df.index.values[0]],
            ["W1", object_info["W1mag"]],
            ["W2", object_info["W2mag"]],
            ["W3", object_info["W3mag"]],
            ["W4", object_info["W4mag"]],
            ["sgscore1", object_info["sgscore1"]],
            ["distpsnr1", object_info["distpsnr1"]],
            ["sgmag1", object_info["sgmag1"]],
            ["srmag1", object_info["srmag1"]],
            ["simag1", object_info["simag1"]],
            ["szmag1", object_info["szmag1"]],
        ],
        columns=["name", "value"],
    )

    astro_object = AstroObject(
        detections=df[df["detected"]],
        forced_photometry=df[~df["detected"]],
        metadata=metadata,
        reference=reference,
    )
    return astro_object


if __name__ == "__main__":
    # Build AstroObjects

    data_dir = "data_241015"
    data_out = data_dir + "_ao"
    lightcurve_filenames = os.listdir(data_dir)
    lightcurve_filenames = [f for f in lightcurve_filenames if "lightcurves_batch" in f]

    object_df = pd.read_parquet(
        os.path.join(data_dir, "objects_with_wise_20241016.parquet")
    )
    object_df.set_index("oid", inplace=True)

    for lc_filename in tqdm(lightcurve_filenames):
        batch_i_str = lc_filename.split(".")[0].split("_")[2]

        lightcurves = pd.read_parquet(os.path.join(data_dir, lc_filename))

        lightcurves.set_index("oid", inplace=True)
        batch_oids = lightcurves.index.unique()

        astro_objects_list = []
        for oid in batch_oids:
            lc = lightcurves.loc[[oid]]
            object_info = object_df.loc[oid]
            try:
                astro_object = create_astro_object(lc, object_info)
                # plot_astro_object(astro_object, unit='diff_flux')
                astro_objects_list.append(astro_object)
            except NoDetections:
                print(object_info)
                print("Object with no detections")

        save_astro_objects_batch(
            astro_objects_list,
            os.path.join(data_out, f"astro_objects_batch_{batch_i_str}.pkl"),
        )
