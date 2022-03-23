from typing import List


class Mapper:
    def __init__(self, step_id_version: str):
        self.step_id_version = step_id_version

    def _convert_object(self, object_: dict) -> dict:
        data = {
            "oid": object_["oid"],
            "ndethist": 1,
            "ncovhist": 1,
            "mjdstarthist": 1,
            "mjdendhist": 1,
            "corrected": 1,
            "stellar": 1,
            "ndet": 1,
            "g_r_max": 1,
            "g_r_max_corr": 1,
            "g_r_mean": 1,
            "g_r_mean_corr": 1,
            "meanra": object_["meanra"],
            "meandec": object_["meandec"],
            "sigmara": 1,  # object_["extra_fields"]["e_ra"],
            "sigmadec": 1,  # object_["extra_fields"]["e_dec"],
            "deltajd": object_["lastmjd"] - object_["firstmjd"],
            "firstmjd": object_["firstmjd"],
            "lastmjd": object_["lastmjd"],
            "step_id_corr": self.step_id_version,
            "diffpos": 1,
            "reference_change": 1,
        }
        return data

    def convert(self, data: List[dict] or dict, model):
        response = None
        if isinstance(data, list):
            if "Object" in str(model):
                response = [self._convert_object(x) for x in data]
            else:
                response = data
        else:
            if "Object" in str(model):
                response = self._convert_object(data)
            else:
                response = data
        return response
