from .base_prv_candidates_strategy import BasePrvCandidatesStrategy
from typing import List
from survey_parser_plugins.core import SurveyParser, id_generator

import pandas as pd

# Keys used on non detections for ZTF
NON_DET_KEYS = ["aid", "tid", "oid", "mjd", "diffmaglim", "fid"]


# Implementation a new parser for PreviousCandidates with SurveyParser signs.
class ZTFPreviousCandidatesParser(SurveyParser):
    _source = "ZTF"
    _generic_alert_message_key_mapping = {
        "candid": "candid",
        "mjd": "jd",
        "fid": "fid",
        "pid": "pid",
        "ra": "ra",
        "dec": "dec",
        "mag": "magpsf",
        "sigmag": "sigmapsf",
        "isdiffpos": "isdiffpos",
        "rb": "rb",
        "rbversion": "rbversion"
    }

    @classmethod
    def parse_message(cls, message: dict) -> dict:
        try:
            oid = message["objectId"]
            prv_candidate = message["candidate"]
            prv_content = cls._generic_alert_message(prv_candidate, cls._generic_alert_message_key_mapping)
            # inclusion of extra attributes
            prv_content["oid"] = oid
            # prv_content["aid"] = id_generator(prv_content["ra"], prv_content["dec"])
            prv_content["aid"] = message["aid"]
            prv_content["tid"] = cls._source
            # attributes modification
            prv_content["mjd"] = prv_content["mjd"] - 2400000.5
            prv_content["isdiffpos"] = 1 if prv_content["isdiffpos"] in ["t", "1"] else -1
            prv_content["parent_candid"] = message["parent_candid"]
            return prv_content
        except KeyError:
            raise KeyError("This parser can't parse message")

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return 'publisher' in message.keys() and cls._source in message["publisher"]

    @classmethod
    def parse(cls, messages: List[dict]) -> List[dict]:
        return list(map(cls.parse_message, messages))


class ZTFPrvCandidatesStrategy(BasePrvCandidatesStrategy):

    def process_prv_candidates(self, alerts: pd.DataFrame):
        detections = {}
        non_detections = []

        for index, alert in alerts.iterrows():
            oid = alert["oid"]
            tid = alert["tid"]
            aid = id_generator(alert["ra"], alert["dec"])
            candid = alert["candid"]
            if alert["extra_fields"]["prv_candidates"] is not None:
                for prv in alert["extra_fields"]["prv_candidates"]:
                    if prv['candid'] is None:
                        prv["aid"] = aid
                        prv["oid"] = oid
                        prv["tid"] = tid
                        non_detections.append(prv)
                    else:
                        detections.update({
                            prv["candid"]: {
                                "objectId": oid,
                                "publisher": tid,
                                "aid": aid,
                                "candidate": prv,
                                "parent_candid": candid
                            }
                        })
                del alert["extra_fields"]["prv_candidates"]
        detections = ZTFPreviousCandidatesParser.parse(list(detections.values()))
        detections = pd.DataFrame(detections)
        non_detections = pd.DataFrame(non_detections) if len(non_detections) else pd.DataFrame()

        if len(non_detections):
            non_detections.rename({"objectId": "oid"}, inplace=True)
            non_detections["mjd"] = non_detections['jd'] - 2400000.5
            non_detections = non_detections[NON_DET_KEYS]
            non_detections = non_detections.drop_duplicates(["oid", "fid", "mjd"])
        return detections, non_detections
