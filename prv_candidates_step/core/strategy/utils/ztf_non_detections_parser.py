from typing import List

from survey_parser_plugins.parsers.ZTFParser import FILTER


class ZTFNonDetectionsParser:
    def parse(self, non_detections: List[dict], aid: str, tid: str, oid: str):
        parsed = []
        for non_detection in non_detections:
            parsed.append(self.parse_non_detection(non_detection, aid, tid, oid))
        return parsed

    def parse_non_detection(self, non_detection: dict, aid: str, tid: str, oid: str):
        return {
            "aid": aid,
            "tid": tid,
            "oid": oid,
            "sid": "ZTF",
            "mjd": self.convert_mjd(non_detection["jd"]),
            "fid": FILTER[non_detection["fid"]],
            "diffmaglim": non_detection["diffmaglim"],
        }

    def convert_mjd(self, jd: float):
        return jd - 2400000.5
