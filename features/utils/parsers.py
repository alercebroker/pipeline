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

    features.replace({np.nan: None}, inplace=True)
    features = features.stack(dropna=False)
    features = features.to_frame()
    features.reset_index(inplace=True)
    features.columns = ["aid", "name", "value"]
    features["fid"] = features.name.apply(get_fid) # quizas tiene que ser lambda
    features["name"] = features.name.apply(check_feature_name)
    features_grouped = features.groupby("aid")
    
    commands_list = [
        {
            "collection": "name",
            "type": "update_features",
            "criteria": {"_id": aid},
            "data": {
                "features_version": features_version,
                "features": features_df.to_dict("records")
            },
            "options": {"upsert": True}        
        } for aid, features_df in features_grouped
    ]
    return commands_list
 
def parse_output(features: pd.DataFrame, alert_data: pd.DataFrame):
    """
    Parse output va a cambiar con la nueva version de compute features
    descripcion pendiente.
    
    """
    output_messages = []
    alert_data.set_index("aid", inplace=True)
    alert_data.drop_duplicates(inplace=True, keep="last")
    for aid, features_oid in features.iterrows():
        features_oid.replace({np.nan: None}, inplace=True)
        message = alert_data.loc[aid]
        features_dict = features_oid.to_dict()
        out_message = {
            "aid": aid,
            "meanra": message["meanra"],
            "meandec": message["meandec"],
            "detections": message["detections"],
            "non_detections": message["non_detections"],
            "xmatches": message["xmatches"],
            "features": features_dict,
        }
        output_messages.extend(out_message)
    
    return output_messages
