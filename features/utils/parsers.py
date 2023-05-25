import numpy as np
import pandas as pd
from features.core._base import BaseFeatureExtractor
from features.core.utils.functions import collapse_fid_columns


def parse_scribe_payload(
    features: pd.DataFrame, extractor_class: BaseFeatureExtractor
):
    """Create the json with the messages for the scribe produccer fron the
    features dataframe. It adds the fid and correct the name.

    :param features: a dataframe that contains a colum with the aid, and
        a column for each feature.
    :param features_version: a string with the features version used
    :return: a list of json with Alerce Scribe commands
    """

    features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)

    commands_list = []
    for aid, features_df in features.iterrows():
        features_list = [
            {"name": name, "fid": None if fid == '' else fid, "value": value}
            for ((name, fid), value) in features_df.items()
        ]
        command = {
            "collection": "name",
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


def parse_output(features: pd.DataFrame, alert_data: list[dict], extractor_class: BaseFeatureExtractor) -> list[dict]:
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
    output_messages = []

    features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
    features.columns = collapse_fid_columns(features, extractor_class.BANDS_MAPPING)

    for message in alert_data:
        aid = message["aid"]
        features_row = features.loc[aid]
        features_dict = features_row.to_dict()
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
