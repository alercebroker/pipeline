import numpy as np
import pandas as pd
from features.utils.features_name_analizer import check_feature_name, get_fid


def parse_scribe_payload(features: pd.DataFrame, features_version: str):
    """Create the json with the messages for the scribe produccer fron the
    features dataframe. It adds the fid and correct the name. th

    :param features: a dataframe that contains a colum with the aid, and
        a column for each feature.
    :param features_version: a string with the features version used
    :return:
    """

    features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)

    commands_list = []
    for aid, features_df in features.iterrows():
        features_list = [
            {"name": name, "fid": None if fid in [0, -99] else fid, "value": value}
            for ((name, fid), value) in features_df.items()
        ]
        command = {
            "collection": "name",
            "type": "update_features",
            "criteria": {"_id": aid},
            "data": {"features_version": features_version, "features": features_list},
            "options": {"upsert": True},
        }
        commands_list.append(command)

    return commands_list


def parse_output(features: pd.DataFrame, alert_data: list[dict]):
    """
    Parse output va a cambiar con la nueva version de compute features
    descripcion pendiente.

    """
    output_messages = []

    features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
    features.columns = features.columns.map(
        lambda lvls: f"{'_'.join(str(l) for l in lvls if l not in [-99, 0, 12])}"
    )

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
