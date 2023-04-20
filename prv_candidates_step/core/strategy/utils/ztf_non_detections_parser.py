from typing import List

from survey_parser_plugins.parsers.ZTFParser import FILTER


class ZTFNonDetectionsParser:
    @classmethod
    def parse(cls, non_detections: List[dict], aid: str, oid: str):
        parsed = []
        for non_detection in non_detections:
            parsed.append(cls.parse_non_detection(non_detection, aid, oid))
        return parsed

    @classmethod
    def parse_non_detection(cls, non_detection: dict, aid: str, oid: str):
        return {
            "aid": aid,
            "tid": "ZTF",
            "oid": oid,
            "sid": "ZTF",
            "mjd": cls.convert_mjd(non_detection["jd"]),
            "fid": FILTER[non_detection["fid"]],
            "diffmaglim": non_detection["diffmaglim"],
        }

    @staticmethod
    def convert_mjd(jd: float):
        return jd - 2400000.5
