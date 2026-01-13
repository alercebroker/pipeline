import json 
import pandas as pd
import numpy as np

def scribe_parser(magstats_list, survey):
    result_messages = []
    for oid, sid_dict in magstats_list.items():
        for sid, stats_data in sid_dict.items():
            stats_data["oid"] = oid
            stats_data["sid"] = sid  
            result_messages.append({
                "step": "magstat",
                "survey": survey,
                "oid": oid,
                "payload": stats_data 
            })
    return result_messages

def remove_timestamp(message: dict):
    message.pop("timestamp", None)
    return message



class NumpyEncoder(json.JSONEncoder):
    """
      Encode data to formats able to be parsed via json dumps for the scribe multisurvey
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):  
            return None
        elif isinstance(obj, pd._libs.missing.NAType): 
            return None
        return super().default(obj)

