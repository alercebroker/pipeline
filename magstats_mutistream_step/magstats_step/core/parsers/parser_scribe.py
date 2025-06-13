import json 
import pandas as pd
import numpy as np

def scribe_parser(magstats_list, survey):
    return [{
        "step": "magstats-ms-step",
        "survey": survey,
        "payload": magstats_list
    }]

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

