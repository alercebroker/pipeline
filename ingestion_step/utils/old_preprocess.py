# Based on https://github.com/alercebroker/correction_step/blob/main/correction/step.py
import pandas as pd
import numpy as np
import warnings
import logging

from ingestion_step.utils.multi_driver.connection import MultiDriverConnection
from ingestion_step.utils.constants import (
    DATAQUALITY_KEYS,
    SS_KEYS,
    REFERENCE_KEYS,
    PS1_KEYS,
    GAIA_KEYS,
    MAGSTATS_TRANSLATE,
    MAGSTATS_UPDATE_KEYS,
)
from typing import List
from lc_correction.compute import (
    apply_mag_stats,
    get_flag_reference,
    get_flag_saturation,
    do_dmdt,
    apply_objstats_from_correction,
    apply_objstats_from_magstats,
)

logger = logging.getLogger("OldPreprocess")


# TEMPORAL CODE
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


def preprocess_dataquality(detections: pd.DataFrame):
    detections = detections[detections["parent_candid"].isna()]
    dataquality = detections.loc[:, detections.columns.isin(DATAQUALITY_KEYS)]
    return dataquality


def get_dataquality(candids: List[int], driver: MultiDriverConnection):
    filter_by = {"candid": {"$in": candids}}
    query = driver.query("Dataquality", engine="psql").find_all(
        filter_by=filter_by, paginate=False
    )
    return pd.DataFrame(query, columns=DATAQUALITY_KEYS)


def insert_dataquality(
    dataquality: pd.DataFrame, driver: MultiDriverConnection
):
    # Not inserting twice
    candids = dataquality["candid"].unique().tolist()
    old_dataquality = get_dataquality(candids, driver)
    already_on_db = dataquality["candid"].isin(old_dataquality["candid"])

    dataquality = dataquality[~already_on_db]
    dataquality.replace({np.nan: None}, inplace=True)
    dict_dataquality = dataquality.to_dict("records")
    driver.query("Dataquality", engine="psql").bulk_insert(dict_dataquality)


def preprocess_ss(
    ss_catalog: pd.DataFrame, detections: pd.DataFrame
) -> pd.DataFrame:
    detections = detections[detections["parent_candid"].isna()]
    if len(ss_catalog):
        oids = ss_catalog["oid"].unique()
    else:
        oids = []
    ss_catalog["new"] = False
    new_metadata = ~detections["oid"].isin(oids)
    new_values = detections.loc[new_metadata, detections.columns.isin(SS_KEYS)]
    if len(new_values) > 0:
        new_values.loc[:, "new"] = True

    return pd.concat([ss_catalog, new_values], ignore_index=True)


def insert_ss(metadata: pd.DataFrame, driver: MultiDriverConnection):
    new_metadata = metadata["new"]
    to_insert = metadata.loc[new_metadata]
    if len(to_insert) > 0:
        to_insert.replace({np.nan: None}, inplace=True)
        dict_to_insert = to_insert.to_dict("records")
        driver.query("Ss_ztf", engine="psql").bulk_insert(dict_to_insert)


def preprocess_reference(metadata: pd.DataFrame, detections: pd.DataFrame):
    detections = detections[detections["parent_candid"].isna()]
    if len(metadata) == 0:
        metadata = pd.DataFrame(columns=REFERENCE_KEYS)
    metadata["new"] = False
    index_metadata = pd.MultiIndex.from_frame(metadata[["oid", "rfid"]])
    index_detections = pd.MultiIndex.from_frame(detections[["oid", "rfid"]])
    already_on_db = index_detections.isin(index_metadata)
    detections["mjdstartref"] = detections["jdstartref"] - 2400000.5
    detections["mjdendref"] = detections["jdendref"] - 2400000.5
    new_values = detections.loc[
        ~already_on_db, detections.columns.isin(REFERENCE_KEYS)
    ]
    if len(new_values) > 0:
        new_values.loc[:, "new"] = True
        new_values.reset_index(inplace=True, drop=True)
    if len(metadata) == 0:
        return new_values
    return pd.concat([metadata, new_values], ignore_index=True)


def insert_reference(metadata: pd.DataFrame, driver: MultiDriverConnection):
    new_metadata = metadata["new"]
    to_insert = metadata[new_metadata]
    if len(to_insert) > 0:
        to_insert.replace({np.nan: None}, inplace=True)
        to_insert = to_insert.astype(object).where(pd.notnull(to_insert), None)
        dict_to_insert = to_insert.to_dict("records")
        driver.query("Reference", engine="psql").bulk_insert(dict_to_insert)


def preprocess_ps1(metadata: pd.DataFrame, detections: pd.DataFrame):
    detections = detections[detections["parent_candid"].isna()]
    if len(metadata) == 0:
        oids = []
    else:
        oids = metadata["oid"].unique()
    metadata["new"] = False
    for i in range(1, 4):
        metadata[f"update{i}"] = False
    new_metadata = ~detections.oid.isin(oids)
    new_values = detections.loc[
        new_metadata, detections.columns.isin(PS1_KEYS)
    ]
    old_values = detections.loc[
        ~new_metadata, detections.columns.isin(PS1_KEYS)
    ]
    if len(new_values) > 0:
        new_values.loc[:, "new"] = True
        new_values.drop_duplicates(["oid"], inplace=True)
        for i in range(1, 4):
            new_values.loc[:, f"unique{i}"] = True
            new_values.loc[:, f"update{i}"] = False
    if len(old_values) > 0:
        join_metadata = old_values.join(
            metadata.set_index("oid"), on="oid", rsuffix="_old"
        )
        for i in range(1, 4):
            difference = join_metadata[
                np.isclose(
                    join_metadata[f"objectidps{i}"],
                    join_metadata[f"objectidps{i}_old"],
                )
                & join_metadata[f"unique{i}"]
            ]
            metadata[f"unique{i}"] = False
            metadata[f"update{i}"] = metadata.oid.isin(difference.oid).astype(
                bool
            )

    data = pd.concat([metadata, new_values], ignore_index=True)
    data["nmtchps"] = data["nmtchps"].astype("int")
    return data


def insert_ps1(metadata: pd.DataFrame, driver: MultiDriverConnection):
    new_metadata = metadata["new"].astype(bool)
    to_insert = metadata[new_metadata]
    to_update = metadata[~new_metadata]

    if len(to_insert) > 0:
        to_insert.replace({np.nan: None}, inplace=True)
        dict_to_insert = to_insert.to_dict("records")
        driver.query("Ps1_ztf", engine="psql").bulk_insert(dict_to_insert)

    if len(to_update) > 0:
        updates = to_update[
            to_update.update1 | to_update.update2 | to_update.update3
        ]
        if len(updates) > 0:
            updates.replace({np.nan: None}, inplace=True)
            oids = updates["oid"].values
            updates = updates[["oid", "unique1", "unique2", "unique3"]]
            dict_updates = updates.to_dict("records")
            filter_by = [{"_id": x} for x in oids]
            driver.query("Ps1_ztf", engine="psql").bulk_update(
                dict_updates, filter_by=filter_by
            )


def preprocess_gaia(
    metadata: pd.DataFrame, detections: pd.DataFrame, tol=1e-03
):
    detections = detections[detections["parent_candid"].isna()]
    if len(metadata) == 0:
        oids = []
    else:
        oids = metadata["oid"].unique()

    metadata[f"update1"] = False
    metadata["new"] = False
    new_metadata = ~detections["oid"].isin(oids)
    new_values = detections.loc[
        new_metadata, detections.columns.isin(GAIA_KEYS)
    ]
    old_values = detections.loc[
        ~new_metadata, detections.columns.isin(GAIA_KEYS)
    ]

    if len(new_values) > 0:
        new_values[f"unique1"] = True
        new_values[f"update1"] = False
        new_values["new"] = True

    if len(old_values) > 0:
        join_metadata = old_values.join(
            metadata.set_index("oid"), on="oid", rsuffix="_old"
        )
        is_the_same_gaia = np.isclose(
            join_metadata["maggaia"].astype("float"),
            join_metadata[f"maggaia_old"].astype("float"),
            rtol=tol,
            atol=tol,
            equal_nan=True,
        ) & np.isclose(
            join_metadata["maggaiabright"].astype("float"),
            join_metadata[f"maggaiabright_old"].astype("float"),
            rtol=tol,
            atol=tol,
            equal_nan=True,
        )
        difference = join_metadata[
            ~(is_the_same_gaia) & join_metadata[f"unique1"]
        ]
        metadata[f"update1"] = metadata.oid.isin(difference.oid).astype(bool)
        metadata[f"unique1"] = False
        metadata["new"] = False

    response = pd.concat([metadata, new_values])
    response = response.astype(object).where(pd.notnull(response), None)
    return response


def insert_gaia(metadata: pd.DataFrame, driver: MultiDriverConnection):
    new_metadata = metadata["new"].astype(bool)
    to_insert = metadata[new_metadata]
    to_update = metadata[~new_metadata]
    if len(to_insert) > 0:
        dict_to_insert = to_insert.to_dict("records")
        driver.query("Gaia_ztf", engine="psql").bulk_insert(dict_to_insert)
    if len(to_update) > 0:
        updates = to_update[to_update.update1]
        if len(updates) > 0:
            updates = updates[["oid", "unique1"]]
            oids = updates["oid"].values
            dict_updates = updates.to_dict("records")
            filter_by = [{"_id": x} for x in oids]
            driver.query("Gaia_ztf", engine="psql").bulk_update(
                dict_updates, filter_by=filter_by
            )


def do_flags(detections: pd.DataFrame, reference: pd.DataFrame):
    diffpos = detections.groupby("oid").apply(lambda x: x.isdiffpos.min() > 0)
    firstmjd = detections.groupby("oid").apply(lambda x: x.mjd.min())
    firstmjd.name = "firstmjd"
    saturation_rate = detections.groupby(["oid", "fid"]).apply(
        get_flag_saturation
    )
    saturation_rate.name = "saturation_rate"

    reference_mjd = reference.join(firstmjd, on="oid")
    reference_change = reference_mjd.groupby("oid").apply(
        lambda x: get_flag_reference(x, x.firstmjd.values[0])
    )

    return (
        pd.DataFrame(
            {"diffpos": diffpos, "reference_change": reference_change}
        ),
        saturation_rate,
    )


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


def do_magstats(
    light_curves: dict,
    magstats: pd.DataFrame,
    ps1: pd.DataFrame,
    reference: pd.DataFrame,
    version: str,
):
    if len(magstats) == 0:
        magstats = pd.DataFrame(columns=["oid", "fid"])
    magstats_index = pd.MultiIndex.from_frame(magstats[["oid", "fid"]])
    detections = light_curves["detections"]
    ps1.set_index("oid", inplace=True)
    reference.set_index(["oid", "rfid"], inplace=True)
    det_ps1 = detections.join(ps1, on="oid", rsuffix="ps1")
    det_ps1_ref = det_ps1.join(reference, on=["oid", "rfid"], rsuffix="ref")
    det_ps1_ref.reset_index(inplace=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        new_magstats = det_ps1_ref.groupby(["oid", "fid"], sort=False).apply(
            apply_mag_stats
        )
    new_magstats.reset_index(inplace=True)
    new_magstats_index = pd.MultiIndex.from_frame(new_magstats[["oid", "fid"]])
    new_magstats["new"] = ~new_magstats_index.isin(magstats_index)
    new_magstats["step_id_corr"] = version
    new_magstats.drop_duplicates(["oid", "fid"], inplace=True)
    reference.reset_index(inplace=True)
    ps1.reset_index(inplace=True)
    return new_magstats


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


def get_last_alert(alerts: pd.DataFrame):
    extra_fields = list(alerts["extra_fields"].values)
    extra_fields = pd.DataFrame(extra_fields, index=alerts.index)
    alerts = alerts.join(extra_fields)
    last_alert = alerts.candid.values.argmax()
    filtered_alerts = alerts.loc[
        :, ["oid", "ndethist", "ncovhist", "jdstarthist", "jdendhist"]
    ]
    last_alert = filtered_alerts.iloc[last_alert]
    return last_alert


def object_stats_df(corrected, magstats, step_name=None, flags=False):
    basic_stats = corrected.groupby("objectId").apply(
        apply_objstats_from_correction, flags=flags
    )
    obj_magstats = []
    for i, g in magstats.groupby("objectId"):
        r = apply_objstats_from_magstats(g)
        r["objectId"] = i
        obj_magstats.append(r)
    obj_magstats = pd.DataFrame(obj_magstats)
    obj_magstats.set_index("objectId", inplace=True)
    basic_stats["step_id_corr"] = (
        "corr_bulk_0.0.1" if step_name is None else step_name
    )
    return basic_stats.join(obj_magstats)


def preprocess_objects_(objects, light_curves, alerts, magstats, version):
    oids = objects.oid.unique()
    extra_fields = list(alerts["extra_fields"].values)
    extra_fields = pd.DataFrame(extra_fields, index=alerts.index)
    alerts = alerts.join(extra_fields)
    alerts = alerts.loc[
        :, ["oid", "ndethist", "ncovhist", "jdstarthist", "jdendhist"]
    ]
    alerts.drop(columns=["oid"], inplace=True)
    detections = light_curves["detections"].drop(
        columns=["ndethist", "ncovhist", "jdstarthist", "jdendhist"]
    )
    detections_last_alert = detections.join(alerts)
    detections_last_alert["objectId"] = detections_last_alert.oid
    detections_last_alert.drop_duplicates(["candid", "oid"], inplace=True)
    detections_last_alert.reset_index(inplace=True)
    magstats["objectId"] = magstats.oid

    new_objects = object_stats_df(
        detections_last_alert, magstats, step_name=version
    )
    new_objects.reset_index(inplace=True)

    new_names = dict(
        [
            (col, col.replace("-", "_"))
            for col in new_objects.columns
            if "-" in col
        ]
    )

    new_objects.rename(columns={"objectId": "oid", **new_names}, inplace=True)
    new_objects["new"] = ~new_objects.oid.isin(oids)
    new_objects["deltajd"] = new_objects["deltamjd"]

    detections_last_alert.drop(columns=["objectId"], inplace=True)
    magstats.drop(columns=["objectId"], inplace=True)

    return new_objects
