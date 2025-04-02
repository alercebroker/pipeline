from typing import Any

import numpy as np
import pandas as pd
import pytest
from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.select_parser import select_parser
from ingestion_step.utils.database import (
    insert_detections,
    insert_forced_photometry,
    insert_non_detections,
    insert_objects,
)

psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}


@pytest.mark.usefixtures("psql_db")
def test_process_alerts_ztf(ztf_alerts: list[dict[str, Any]], psql_db: PsqlDatabase):
    messages = ztf_alerts

    parser = select_parser("ztf")

    parsed_data = parser.parse(messages)

    insert_objects(psql_db, parsed_data["objects"])
    insert_detections(psql_db, parsed_data["detections"])
    insert_non_detections(psql_db, parsed_data["non_detections"])
    insert_forced_photometry(psql_db, parsed_data["forced_photometries"])

    result = parsed_data

    def groupby_message_id(
        df: pd.DataFrame,
    ) -> dict[int, list[dict[str, Any]]]:
        return (
            df.groupby("message_id")
            .apply(lambda x: x.to_dict("records"), include_groups=False)
            .to_dict()
        )

    det_columns = [
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

    objects = result["objects"].replace({np.nan: None})
    detections = pd.concat(
        [result["detections"], result["forced_photometries"]]
    ).replace({np.nan: None})
    non_detections = result["non_detections"].replace({np.nan: None})

    detections["extra_fields"] = detections[
        detections.columns.difference(det_columns)
    ].to_dict("records")
    detections = detections[det_columns]

    message_objects = groupby_message_id(objects)
    message_detections = groupby_message_id(detections)
    message_non_detections = groupby_message_id(non_detections)

    messages = []
    for message_id, objects in message_objects.items():
        detections = message_detections.get(message_id, [])
        non_detections = message_non_detections.get(message_id, [])

        assert len(objects) == 1
        obj = objects[0]

        messages.append(
            {
                "oid": obj["oid"],
                "candid": obj["measurement_id"],
                "detections": detections,
                "non_detections": non_detections,
            }
        )

    print(messages)
    assert False
