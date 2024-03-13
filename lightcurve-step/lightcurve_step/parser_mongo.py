from pymongo.cursor import Cursor
from .parser_utils import get_fid


def parse_mongo_forced_photometry(forced_photometries: Cursor):
    parsed_forced_photometry = list(forced_photometries)
    return parsed_forced_photometry


def parse_mongo_detection(detections: Cursor):
    parsed_detections = list(detections)
    parsed_detections = [
        # assign candid as field candid if present, else assign _id as candid
        # to comply with the legacy schema
        {
            **det,
            "fid": get_fid(det["fid"]),
        }
        for det in parsed_detections
    ]
    return parsed_detections


def parse_mongo_non_detection(non_detections: Cursor):
    parsed_non_detections = list(non_detections)
    return parsed_non_detections
