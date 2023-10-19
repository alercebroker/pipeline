import logging
import numpy as np
import pandas as pd
from features.core.utils.functions import collapse_fid_columns


def parse_scribe_payload(
    messages_aid_oid: dict, features: pd.DataFrame, extractor_class
):
    """Create the json with the messages for the scribe produccer fron the
    features dataframe. It adds the fid and correct the name.

    :param messages_aid_oid: a dict that cointains list of oids for each
        aid in the step input.
    :param features: a dataframe that contains a colum with the aid, and
        a column for each feature.
    :param features_version: a string with the features version used
    :return: a list of json with Alerce Scribe commands
    """

    features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
    if extractor_class.NAME == "ztf_lc_features":
        return _parse_scribe_payload_ztf(
            messages_aid_oid, features, extractor_class
        )
    if extractor_class.NAME == "elasticc_lc_features":
        return _parse_scribe_payload_elasticc(features, extractor_class)
    else:
        raise Exception(
            'Cannot parse scribe payload for extractor "{}"'.format(
                extractor_class.NAME
            )
        )


def _parse_scribe_payload_elasticc(features, extractor_class):
    def get_fid(feature_name: str):
        band = feature_name.split("_")[-1]
        if band in ["u", "g", "r", "i", "z", "Y"]:
            return band
        return None

    commands_list = []
    for aid, features_df in features.iterrows():
        features_list = [
            {"name": name, "fid": get_fid(name), "value": value}
            for (name, value) in features_df.items()
        ]
        command = {
            "collection": "object",
            "type": "update_features",
            "criteria": {"_id": aid},
            "data": {
                "features_version": extractor_class.VERSION,
                "features_group": extractor_class.NAME,
                "features": features_list,
            },
            "options": {"upsert": True},
        }
        commands_list.append(command)

    return commands_list


def _parse_scribe_payload_ztf(
    messages_aid_oid: dict, features, extractor_class
):
    commands_list = []
    for aid, features_df in features.iterrows():
        FID_MAP = {"g": 1, "r": 2, "gr": 12, "rg": 12}
        features_list = [
            {"name": name, "fid": 0 if fid == "" else FID_MAP[fid], "value": value}
            for ((name, fid), value) in features_df.items()
        ]
        command = {
            "collection": "object",
            "type": "update_features",
            "criteria": {"_id": aid, "oid": messages_aid_oid[aid]},
            "data": {
                "features_version": extractor_class.VERSION,
                "features_group": extractor_class.NAME,
                "features": features_list,
            },
            "options": {"upsert": True},
        }
        commands_list.append(command)

    return commands_list


def parse_output(
    features: pd.DataFrame,
    alert_data: list[dict],
    extractor_class,
) -> list[dict]:
    """
    Parse output of the step. It uses the input data to extend the schema to
    add the features of each object, identified by its aid.

    :param features: a dataframe with the calculated features, with a column with
        the aid and a colum for each feature (with 2 levels one for the feature name
        the next with the band of the feature calculated)
    :param alert_data: the imput for the step
    :returnn: a list of dictiories, each input object with its data and the
        features calculated.

    """
    if extractor_class.NAME == "ztf_lc_features":
        return _parse_output_ztf(features, alert_data, extractor_class)
    elif extractor_class.NAME == "elasticc_lc_features":
        return _parse_output_elasticc(features, alert_data, extractor_class)
    else:
        raise Exception(
            'Cannot parse output for extractor "{}"'.format(
                extractor_class.NAME
            )
        )


def _parse_output_elasticc(features, alert_data, extractor_class):
    output_messages = []
    if len(features):
        features.replace(
            {np.nan: None, np.inf: None, -np.inf: None}, inplace=True
        )
    for message in alert_data:
        aid = message["aid"]
        try:
            features_dict = features.loc[aid].to_dict()
        except KeyError:  # No feature for the object
            logger = logging.getLogger("alerce")
            logger.info("Could not calculate features of object %s", aid)
            features_dict = None
        out_message = {
            "aid": aid,
            "meanra": message["meanra"],
            "meandec": message["meandec"],
            "detections": message["detections"],
            "non_detections": message["non_detections"],
            "xmatches": message["xmatches"],
            "features": features_dict,
        }
        output_messages.append(out_message)

    return output_messages


def _parse_output_ztf(features, alert_data, extractor_class):
    output_messages = []

    if len(features):
        features.replace(
            {np.nan: None, np.inf: None, -np.inf: None}, inplace=True
        )
        features = collapse_fid_columns(
            features, extractor_class.BANDS_MAPPING
        )

    for message in alert_data:
        aid = message["aid"]
        try:
            features_dict = features.loc[aid].to_dict()
        except KeyError:  # No feature for the object
            logger = logging.getLogger("alerce")
            logger.info("Could not calculate features of object %s", aid)
            features_dict = None
        out_message = {
            "aid": aid,
            "meanra": message["meanra"],
            "meandec": message["meandec"],
            "detections": message["detections"],
            "non_detections": message["non_detections"],
            "xmatches": message["xmatches"],
            "features": features_dict,
        }
        output_messages.append(out_message)

    return output_messages
