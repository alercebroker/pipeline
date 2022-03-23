# Based on https://github.com/alercebroker/correction_step/blob/main/correction/step.py
import pandas as pd
import numpy as np

from ingestion_step.utils.multi_driver.connection import MultiDriverConnection
from ingestion_step.utils.constants import DATAQUALITY_KEYS, SS_KEYS, REFERENCE_KEYS, PS1_KEYS
from typing import List


# TEMPORAL CODE
def get_catalog(aids: List[str or int], table: str, driver: MultiDriverConnection):
    filter_by = {"aid": {"$in": aids}}
    references = driver.query(table, engine="psql").find_all(filter_by=filter_by, paginate=False)
    return pd.DataFrame(references)


def preprocess_dataquality(detections: pd.DataFrame):
    dataquality = detections.loc[:, detections.columns.isin(DATAQUALITY_KEYS)]
    return dataquality


def get_dataquality(candids: List[int], driver: MultiDriverConnection):
    filter_by = {"candid": {"$in": candids}}
    query = driver.query("Dataquality", engine="psql").find_all(filter_by=filter_by, paginate=False)
    return pd.DataFrame(query, columns=DATAQUALITY_KEYS)


def insert_dataquality(dataquality: pd.DataFrame, driver: MultiDriverConnection):
    # Not inserting twice
    candids = dataquality["candid"].unique().tolist()
    old_dataquality = get_dataquality(candids, driver)
    already_on_db = dataquality["candid"].isin(old_dataquality["candid"])

    dataquality = dataquality[~already_on_db]
    dataquality.replace({np.nan: None}, inplace=True)
    dict_dataquality = dataquality.to_dict("records")
    driver.query("Dataquality", engine="psql").bulk_insert(dict_dataquality)


def preprocess_ss(ss_catalog: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
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
        new_values.replace({np.nan: None}, inplace=True)
    if len(metadata) == 0:
        return new_values
    return pd.concat([metadata, new_values], ignore_index=True)


def insert_reference(metadata: pd.DataFrame, driver: MultiDriverConnection):
    new_metadata = metadata["new"]
    to_insert = metadata[new_metadata]
    if len(to_insert) > 0:
        dict_to_insert = to_insert.to_dict("records")
        driver.query("Reference", engine="psql").bulk_insert(dict_to_insert)


def preprocess_ps1(metadata: pd.DataFrame, detections: pd.DataFrame):
    if len(metadata) == 0:
        oids = []
    else:
        oids = metadata["oid"].unique()
    metadata["new"] = False
    for i in range(1, 4):
        metadata[f"update{i}"] = False
    new_metadata = ~detections.oid.isin(oids)
    new_values = detections.loc[new_metadata, detections.columns.isin(PS1_KEYS)]
    old_values = detections.loc[~new_metadata, detections.columns.isin(PS1_KEYS)]
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
            metadata[f"update{i}"] = metadata.oid.isin(difference.oid).astype(bool)

    return pd.concat([metadata, new_values], ignore_index=True)


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
            driver.query("Ps1_ztf", engine="psql").bulk_update(dict_updates, filter_by=filter_by)
