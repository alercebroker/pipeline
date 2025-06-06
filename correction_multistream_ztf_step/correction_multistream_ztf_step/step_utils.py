
import pandas as pd
import numpy as np

from core.parsers.parser_utils import parse_output
from core.corrector import Corrector

from core.DB.database_sql import (
    _get_sql_detections,
    _get_sql_forced_photometries,
    _get_sql_non_detections,
)

from core.parsers.parser_sql import (
    parse_sql_detection,
    parse_sql_forced_photometry,
    parse_sql_non_detection,
)


def split_dets(msg: list, all_detections: list, oid) -> list:

    for detection in msg["detections"]:
        parsed_detection = detection.copy()
        parsed_detection["oid"] = oid
        parsed_detection["new"] = True
        keys_to_remove = [
            "magpsf",
            "magpsf_corr",
            "sigmapsf",
            "sigmapsf_corr",
            "sigmapsf_corr_ext",
        ]

        #   Remove the keys from the extra_fields dictionary (sigmapsf is duped with mag_corr and it doesnt make sense having that data in the message)
        for key in keys_to_remove:
            if key in parsed_detection["extra_fields"]:
                parsed_detection["extra_fields"].pop(key, None)

        keys_to_remove = [
            "magpsf",
            "magpsf_corr",
            "sigmapsf",
            "sigmapsf_corr",
            "sigmapsf_corr_ext",
        ]

        #   Remove the keys from the extra_fields dictionary (sigmapsf is duped with mag_corr and it doesnt make sense having that data in the message)
        for key in keys_to_remove:
            if key in parsed_detection["extra_fields"]:
                parsed_detection["extra_fields"].pop(key, None)

        all_detections.append(parsed_detection)

    return all_detections

def split_nondets(msg: list, all_non_detections: list, oid) -> list:

    for non_detection in msg["non_detections"]:
        parsed_non_detection = non_detection.copy()
        parsed_non_detection["oid"] = oid
        all_non_detections.append(parsed_non_detection)

    return all_non_detections

def split_det_nondets(messages: list[dict]) -> tuple[list, list, pd.DataFrame]:

    all_detections = []
    all_non_detections = []
    msg_data = []

    for msg in messages:
        oid = msg["oid"]

        measurement_id = msg["measurement_id"]
        msg_data.append({"oid": oid, "measurement_id": measurement_id})

        all_detections = split_dets(msg, all_detections, oid)
        all_non_detections = split_nondets(msg, all_non_detections, oid)

    msg_df = pd.DataFrame(msg_data)

    return all_detections, all_non_detections, msg_df

def det_to_df(all_detections) -> pd.DataFrame:
     
    detections_df = pd.DataFrame(
        all_detections
    )  # We will always have detections BUT not always non_detections
    # Keep the parent candid as int instead of scientific notation
    detections_df["parent_candid"] = detections_df["parent_candid"].astype("Int64")
    # Tranform the NA parent candid to None
    detections_df["parent_candid"] = (
        detections_df["parent_candid"]
        .astype(object)
        .where(~detections_df["parent_candid"].isna(), None)
    )
    # Keep the parent candid as int instead of scientific notation
    detections_df["parent_candid"] = detections_df["parent_candid"].astype("Int64")
    # Tranform the NA parent candid to None
    detections_df["parent_candid"] = (
        detections_df["parent_candid"]
        .astype(object)
        .where(~detections_df["parent_candid"].isna(), None)
    )

    return detections_df

def nondets_to_df(all_non_detections) -> pd.DataFrame:

    if all_non_detections:
        non_detections_df = pd.DataFrame(all_non_detections)
    else:
        non_detections_df = pd.DataFrame(
            columns=["oid", "measurement_id", "band", "mjd", "diffmaglim"]
        )

    return non_detections_df

def non_det_nan_replace(non_detections: pd.DataFrame):

    non_detections = (
        non_detections.replace(np.nan, None)
        if not non_detections.empty
        else pd.DataFrame(columns=["oid"])
    )

    return non_detections

def process_messages(messages: list[dict]) -> dict:
    all_detections, all_non_detections, msg_df = split_det_nondets(messages)
    detections_df = det_to_df(all_detections)
    non_detections_df = nondets_to_df(all_non_detections)
    
    last_mjds = detections_df.groupby("oid")["mjd"].max().to_dict()
    oids = list(set(msg_df["oid"].unique()))
    
    return {
        'detections': detections_df.to_dict("records"),
        'non_detections': non_detections_df.to_dict("records"),
        'last_mjds': last_mjds,
        'oids': oids,
        'msg_df': msg_df
    }

def fetch_database_data(oids: list, db_sql) -> dict:
    db_detections = _get_sql_detections(oids, db_sql, parse_sql_detection)
    db_non_detections = _get_sql_non_detections(oids, db_sql, parse_sql_non_detection)
    db_forced = _get_sql_forced_photometries(oids, db_sql, parse_sql_forced_photometry)
    
    return {
        'detections': db_detections,
        'non_detections': db_non_detections,
        'forced_photometries': db_forced
    }

def merge_and_clean_data(processed_data: dict, db_data: dict) -> dict:
    detections = pd.DataFrame(
        processed_data['detections'] + db_data['detections'] + db_data['forced_photometries']
    )
    non_detections = pd.DataFrame(processed_data['non_detections'] + db_data['non_detections'])
    
    detections["measurement_id"] = detections["measurement_id"].astype(str)
    detections = detections.sort_values(["has_stamp", "new"], ascending=[False, False])
    detections = detections.drop_duplicates(["measurement_id", "oid"], keep="first")
    non_detections = non_detections.drop_duplicates(["oid", "band", "mjd"])
    
    return {
        'detections': detections,
        'non_detections': non_detections,
        'last_mjds': processed_data['last_mjds']
    }

def apply_corrections(merged_data: dict, config: dict) -> dict:
    detections = merged_data['detections']
    non_detections = non_det_nan_replace(merged_data['non_detections'])
    
    if not config["FEATURE_FLAGS"].get("SKIP_MJD_FILTER", False):
        detections = detections[detections["mjd"] <= detections["oid"].map(merged_data['last_mjds'])]
    
    corrector = Corrector(detections)
    corrected_detections = corrector.corrected_as_records()
    coords = corrector.coordinates_as_records()
    non_detections = non_detections.replace({float("nan"): None})
    non_detections = non_detections.drop_duplicates(["oid", "band", "mjd"])
    
    return {
        'detections': corrected_detections,
        'non_detections': non_detections.to_dict("records"),
        'coords': coords
    }

def build_result(corrected_data: dict, msg_df: pd.DataFrame) -> dict:
    measurement_ids = (
        msg_df.groupby("oid")["measurement_id"]
        .apply(lambda x: [str(id) for id in x])
        .to_dict()
    )
    
    result = {
        "detections": corrected_data['detections'],
        "non_detections": corrected_data['non_detections'],
        "coords": corrected_data['coords'],
        "measurement_ids": measurement_ids,
    }
    
    return parse_output(result)