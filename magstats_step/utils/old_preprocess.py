import pandas as pd
import numpy as np
import warnings

# Should make this a separate package
from magstats_step.utils.multi_driver.connection import MultiDriverConnection

from magstats_step.utils.constants import (
    MAGSTATS_TRANSLATE,
    MAGSTATS_UPDATE_KEYS,
)

from typing import List

from lc_correction.compute import (
    apply_mag_stats,
    do_dmdt)

# Temporal, why?
def get_catalog(
    aids: List[str or int], table: str, driver: MultiDriverConnection
):
    filter_by = {"aid": {"$in": aids}}
    catalog = driver.query(table, engine="psql").find_all(
        filter_by=filter_by, paginate=False
    )
    catalog = pd.DataFrame(catalog)
    catalog.replace({np.nan: None}, inplace=True)
    return catalog

def do_dmdt_df(magstats, non_dets, dt_min=0.5):
    magstats.set_index(["objectId", "fid"], inplace=True, drop=True)
    non_dets_magstats = non_dets.join(
        magstats, on=["objectId", "fid"], how="inner", rsuffix="_stats"
    )
    responses = []
    for i, g in non_dets_magstats.groupby(["objectId", "fid"]):
        response = do_dmdt(g)
        response["oid"] = i[0]
        response["fid"] = i[1]
        responses.append(response)
    result = pd.DataFrame(responses)
    magstats.reset_index(inplace=True)
    return result


def compute_dmdt(light_curves: dict, magstats: pd.DataFrame):
    if len(light_curves["non_detections"]) == 0:
        return pd.DataFrame()
    non_detections = light_curves["non_detections"]
    non_detections["objectId"] = non_detections["oid"]
    if "index" in non_detections.columns:
        non_detections.drop(columns=["index"], inplace=True)
        non_detections.reset_index(drop=True, inplace=True)
    magstats["objectId"] = magstats["oid"]
    dmdt = do_dmdt_df(magstats, non_detections)
    non_detections.drop(columns=["objectId"], inplace=True)
    magstats.drop(columns=["objectId"], inplace=True)
    return dmdt

def insert_magstats(magstats: pd.DataFrame, driver: MultiDriverConnection):
    new_magstats = magstats["new"].astype(bool)
    to_insert = magstats[new_magstats]
    to_update = magstats[~new_magstats]

    if len(to_insert) > 0:
        to_insert.replace({np.nan: None}, inplace=True)
        dict_to_insert = to_insert.to_dict("records")
        driver.query("MagStats", engine="psql").bulk_insert(dict_to_insert)

    if len(to_update) > 0:
        to_update.replace({np.nan: None}, inplace=True)
        to_update.rename(
            columns={**MAGSTATS_TRANSLATE},
            inplace=True,
        )
        filter_by = [
            {"_id": x["oid"], "fid": x["fid"]} for i, x in to_update.iterrows()
        ]
        to_update = to_update[["oid", "fid"] + MAGSTATS_UPDATE_KEYS]
        dict_to_update = to_update.to_dict("records")
        driver.query("MagStats", engine="psql").bulk_update(
            dict_to_update, filter_by=filter_by
        )
