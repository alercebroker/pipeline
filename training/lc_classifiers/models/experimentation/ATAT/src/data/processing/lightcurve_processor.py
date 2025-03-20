import pandas as pd
import numpy as np
from src.data.processing.lightcurve_utils import (
    separate_by_filter,
    normalizing_time,
    create_mask,
    mask_detection,
    normalizing_time_detection,
    mask_photometry,
    normalizing_time_photometric
)

def processing_lightcurve(lightcurve, dict_info, dict_cols):
    if dict_info["type_windows"] == "windows":
        lightcurve["window_id"] = lightcurve.index // dict_info["max_obs"]
        lightcurve = lightcurve.groupby(['oid', "window_id"]).agg(lambda x: list(x))

    elif dict_info["type_windows"] == "max":
        dict_info["max_obs"] = lightcurve["band"].value_counts().max()
        lightcurve = lightcurve.groupby(['oid']).agg(lambda x: list(x))

    else:
        lightcurve = lightcurve.groupby(['oid']).agg(lambda x: list(x))

    pd_dataset = pd.DataFrame()
    for col_name in dict_cols.values():
        if col_name not in ["oid", "band", 'class_name']:
            pd_dataset[col_name] = lightcurve.apply(
                lambda x: separate_by_filter(
                    x[col_name],
                    x["band"],
                    x["time"],
                    x["detected"],
                    dict_info,
                ),
                axis=1,
            )

    pd_dataset["time"] = pd_dataset.apply(
        lambda x: normalizing_time(x["time"]), axis=1
    )
    pd_dataset["mask"] = pd_dataset.apply(lambda x: create_mask(x["brightness"]), axis=1)

    pd_dataset["mask_detection"] = pd_dataset.apply(
        lambda x: mask_detection(x["mask"], x["detected"]), axis=1
    )
    pd_dataset["time_detection"] = pd_dataset.apply(
        lambda x: normalizing_time_detection(x["time"], x["mask_detection"]), axis=1
    )

    pd_dataset["mask_photometry"] = pd_dataset.apply(
        lambda x: mask_photometry(x["mask"], x["detected"]), axis=1
    )
    pd_dataset["time_photometry"] = pd_dataset.apply(
        lambda x: normalizing_time_photometric(x["time"], x["mask_photometry"]),
        axis=1,
    )
    return pd_dataset.reset_index()
