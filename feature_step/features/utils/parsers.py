import logging
import numpy as np
import pandas as pd
from lc_classifier.base import AstroObject, query_ao_table
from features.core.utils.functions import collapse_fid_columns
from typing import List, Dict


def detections_to_astro_objects(detections: List[Dict], xmatches: Dict) -> AstroObject:
    detection_keys = [
        "oid",
        "candid",
        "aid",
        "tid",
        "sid",
        "pid",
        "ra",
        "dec",
        "mjd",
        "mag_corr",
        "e_mag_corr",
        "fid",
        "isdiffpos",
        "forced",
        # 'sgscore1'
    ]

    values = []
    for detection in detections:
        values.append([detection[key] for key in detection_keys])

    a = pd.DataFrame(data=values, columns=detection_keys)
    a.rename(
        columns={"mag_corr": "brightness", "e_mag_corr": "e_brightness"}, inplace=True
    )
    a["unit"] = "magnitude"
    a_flux = a.copy()
    a_flux["brightness"] = 10.0 ** (-0.4 * (a["brightness"] - 23.9)) * a["isdiffpos"]
    a_flux["e_brightness"] *= 0.4 * np.log(10) * np.abs(a_flux["brightness"])
    a_flux["unit"] = "diff_flux"
    a = pd.concat([a, a_flux], axis=0)
    a.set_index("aid", inplace=True)

    aid = a.index.values[0]
    oid = a["oid"].iloc[0]

    aid_forced = a[a["forced"]]
    aid_detections = a[~a["forced"]]

    metadata = pd.DataFrame(
        [
            ["aid", aid],
            ["oid", oid],
            ["W1", xmatches["allwise"]["W1mag"]],
            ["W2", xmatches["allwise"]["W2mag"]],
            ["W3", xmatches["allwise"]["W3mag"]],
            ["W4", xmatches["allwise"]["W4mag"]],
            ["sgscore1", detections[0]["extra_fields"]["sgscore1"]],
            ["sgmag1", detections[0]["extra_fields"]["sgmag1"]],
            ["srmag1", detections[0]["extra_fields"]["srmag1"]],
            ["distpsnr1", detections[0]["extra_fields"]["distpsnr1"]],
        ],
        columns=["name", "value"],
    )

    astro_object = AstroObject(
        detections=aid_detections, forced_photometry=aid_forced, metadata=metadata
    )
    return astro_object


def parse_scribe_payload(
    astro_objects: List[AstroObject], features_version, features_group
):
    """Create the json with the messages for the scribe producer from the
    features dataframe. It adds the fid and correct the name.

    :param astro_objects: a list of AstroObjects with computed features inside.
    :param features_version: a string with the features version used
    :return: a list of json with Alerce Scribe commands
    """

    def get_fid(band: str):
        """
        Parses the number used to reference the fid in the ztf alerts
        to the string value corresponding
        """
        fid_map = {"g": 1, "r": 2, "g,r": 12}
        if band in fid_map:
            return fid_map[band]
        return 0

    # features = features.replace({np.nan: None, np.inf: None, -np.inf: None})
    commands_list = []

    for astro_object in astro_objects:
        ao_features = astro_object.features[["name", "fid", "value"]].copy()
        ao_features["fid"] = ao_features["fid"].apply(get_fid)
        ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
        oid = query_ao_table(astro_object.metadata, "oid")

        features_list = ao_features.to_dict("records")

        command = {
            "collection": "object",
            "type": "update_features",
            "criteria": {"_id": oid},
            "data": {
                "features_version": features_version,
                "features_group": features_group,
                "features": features_list,
            },
            "options": {"upsert": True},
        }
        commands_list.append(command)

    return commands_list


def parse_output(
    astro_objects: List[AstroObject], messages: List[Dict], candids: Dict
) -> list[dict]:
    """
    Parse output of the step. It uses the input data to extend the schema to
    add the features of each object, identified by its oid.
    astro_objects and messages must be in the same order

    :param astro_objects:
    :param messages:
    :param candids:
    :return: a list of dictionaries, each input object with its data and the
        features calculated.
    """

    output_messages = []
    for message, astro_object in zip(messages, astro_objects):
        oid = message["oid"]
        candid = candids[oid]

        ao_features = astro_object.features[["name", "fid", "value"]].copy()
        fid_map = {"g": "_1", "r": "_2", "g,r": "_12", None: ""}
        ao_features["name"] += ao_features["fid"].map(fid_map)
        ao_features = ao_features.sort_values("name")
        ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
        oid_ao = query_ao_table(astro_object.metadata, "oid")
        assert oid_ao == oid
        feature_names = [f.replace("-", "_") for f in ao_features["name"].values]

        features_for_oid = dict(
            zip(feature_names, ao_features["value"].astype(np.double))
        )
        out_message = {
            "oid": oid,
            "candid": candid,
            "detections": message["detections"],
            "non_detections": message["non_detections"],
            "xmatches": message["xmatches"],
            "features": features_for_oid,
        }
        output_messages.append(out_message)

    return output_messages
