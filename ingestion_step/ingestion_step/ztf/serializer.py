from typing import Any

import numpy as np
import pandas as pd

from ingestion_step.core.parser_interface import ParsedData


def serialize_object(objects: pd.DataFrame):
    objs = objects.replace({np.nan: None})

    return objs


def serialize_detections(
    detections: pd.DataFrame, forced_photometries: pd.DataFrame
):
    dets = pd.concat([detections, forced_photometries])
    dets = dets.replace({np.nan: None})

    needed_columns = [
        "message_id",
        "oid",
        "sid",
        "pid",
        "tid",
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

    extra_fields_cols = dets.columns.difference(needed_columns)
    dets["extra_fields"] = dets[extra_fields_cols].to_dict("records")
    dets = dets[needed_columns]

    return dets


def serialize_non_detections(non_detections: pd.DataFrame):
    non_dets = non_detections.replace({np.nan: None})

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


def groupby_messageid(df: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    return (
        df.groupby("message_id")
        .apply(lambda x: x.to_dict("records"), include_groups=False)
        .to_dict()
    )


def serialize_ztf(data: ParsedData) -> list[dict[str, Any]]:
    objects = serialize_object(data["objects"])
    detections = serialize_detections(
        data["detections"], data["forced_photometries"]
    )
    non_detections = serialize_non_detections(data["non_detections"])

    message_objects = groupby_messageid(objects)
    message_detections = groupby_messageid(detections)
    message_non_detections = groupby_messageid(non_detections)

    messages = []
    for message_id, objects in message_objects.items():
        detections = message_detections.get(message_id, [])
        non_detections = message_non_detections.get(message_id, [])

        assert len(objects) == 1
        obj = objects[0]

        messages.append(
            {
                "oid": obj["oid"],
                "measurement_id": obj["measurement_id"],
                "detections": detections,
                "non_detections": non_detections,
            }
        )

    return messages
