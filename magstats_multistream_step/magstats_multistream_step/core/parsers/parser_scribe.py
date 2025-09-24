import json 
import pandas as pd
import numpy as np




def scribe_parser(magstats_list, survey):
    result_messages = []
    for oid, magstats_data in magstats_list.items():
        magstats_data["oid"] = oid
        result_messages.append({
        "step": "magstat",
        "survey": survey,
        "oid": oid,
        "payload": magstats_data
        })
    return result_messages


def scribe_parser_objects(objects_list, survey):
    result_messages_object = []
    for objects_data in objects_list:
        result_messages_object.append({
            "step": "magstat_objects", 
            "survey": survey,
            "oid": objects_data["oid"],
            "payload": objects_data
        })
    return result_messages_object

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

