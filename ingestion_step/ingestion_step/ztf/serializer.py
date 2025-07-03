import pandas as pd


def serialize_detections(detections: pd.DataFrame):
    # Bad fix for precision loss in kafka
    # for bad_column in ["objectidps1", "objectidps2", "objectidps3", "tblid"]:
    #     detections[bad_column] = detections[bad_column].astype(pd.StringDtype())

    needed_columns = [
        "message_id",
        "oid",
        "sid",
        "tid",
        "pid",
        "band",
        "measurement_id",
        "mjd",
        "ra",
        "e_ra",
        "dec",
        "e_dec",
        "mag",
        "e_mag",
        "isdiffpos",
        "has_stamp",
        "forced",
        "parent_candid",
        "extra_fields",
    ]

    extra_fields_cols = detections.columns.difference(needed_columns)
    detections["extra_fields"] = detections[extra_fields_cols].to_dict("records")
    dets = detections[needed_columns]

    return dets


def serialize_non_detections(non_detections: pd.DataFrame):
    non_dets = non_detections
    needed_columns = [
        "message_id",
        "oid",
        "sid",
        "tid",
        "band",
        "mjd",
        "diffmaglim",
    ]

    non_dets = non_dets[needed_columns]

    return non_dets
