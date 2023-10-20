import copy
import numpy as np
import pickle
from dataclasses import asdict

from survey_parser_plugins.core import SurveyParser
from survey_parser_plugins.core.generic import GenericNonDetection
from survey_parser_plugins.core.mapper import Mapper
from survey_parser_plugins.parsers import ZTFParser


def prv_non_detections_mapper() -> dict:
    mapping = copy.deepcopy(ZTFParser._mapping)
    preserve = ["oid", "sid", "tid", "fid", "mjd"]

    mapping = {k: v for k, v in mapping.items() if k in preserve}
    mapping.update({"diffmaglim": Mapper(origin="diffmaglim")})
    return mapping


def prv_forced_photometry_mapper() -> dict:
    mapping = copy.deepcopy(ZTFParser._mapping)

    mapping.update(
        {
            "e_ra": Mapper(lambda: 0),
            "e_dec": Mapper(lambda: 0),
            "isdiffpos": Mapper(
                lambda x: 1 if x >= 0 else -1, origin="forcediffimflux"
            ),
        }
    )

    return mapping


class ZTFPreviousDetectionsParser(SurveyParser):
    _mapping = ZTFParser._mapping

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(ZTFPreviousDetectionsParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict, oid: str) -> dict:
        message = message.copy()
        message["objectId"] = oid
        return asdict(cls.parse_message(message))


class ZTFForcedPhotometryParser(SurveyParser):
    _mapping = prv_forced_photometry_mapper()

    @classmethod
    def __calculate_mag(cls, data):
        magzpsci = data["magzpsci"]
        flux2uJy = 10.0 ** ((8.9 - magzpsci) / 2.5) * 1.0e6

        forcediffimflux = data["forcediffimflux"] * flux2uJy
        forcediffimfluxunc = data["forcediffimfluxunc"] * flux2uJy

        mag = -2.5 * np.log10(np.abs(forcediffimflux)) + 23.9
        e_mag = 1.0857 * forcediffimfluxunc / np.abs(forcediffimflux)
        return mag, e_mag

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(ZTFForcedPhotometryParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict, candid: str, oid: str, ra: float, dec: float) -> dict:
        message = message.copy()
        message["candid"] = candid
        message["magpsf"], message["sigmapsf"] = cls.__calculate_mag(message)
        message["objectId"] = oid

        message["ra"] = ra
        message["dec"] = dec
        return asdict(cls.parse_message(message))


class ZTFNonDetectionsParser(SurveyParser):
    _mapping = prv_non_detections_mapper()
    _Model = GenericNonDetection

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(ZTFNonDetectionsParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse(cls, message: dict, oid: str) -> dict:
        message = message.copy()
        message["objectId"] = oid
        return asdict(cls.parse_message(message))


def extract_detections_and_non_detections(alert: dict) -> dict:
    detections = [alert]
    non_detections = []

    prv_candidates = alert["extra_fields"].pop("prv_candidates")
    prv_candidates = pickle.loads(prv_candidates) if prv_candidates else []

    aid, oid, parent = alert["aid"], alert["oid"], alert["candid"]
    for candidate in prv_candidates:
        if candidate["candid"]:
            candidate = ZTFPreviousDetectionsParser.parse(candidate, oid)
            candidate.update(
                {
                    "aid": aid,
                    "has_stamp": False,
                    "forced": False,
                    "parent_candid": parent,
                    "extra_fields": {
                        **alert["extra_fields"],
                        **candidate["extra_fields"],
                    },
                }
            )
            candidate.pop("stamps", None)
            detections.append(candidate)
        else:
            candidate = ZTFNonDetectionsParser.parse(candidate, oid)
            candidate.update({"aid": aid})
            candidate.pop("stamps", None)
            candidate.pop("extra_fields", None)
            non_detections.append(candidate)

    alert["extra_fields"]["parent_candid"] = None

    prv_forced_photometries = alert["extra_fields"].pop("fp_hists", None)
    if prv_forced_photometries:
        prv_forced_photometries = pickle.loads(prv_forced_photometries)
    else:
        prv_forced_photometries = []

    for fp in prv_forced_photometries:
        # concat parent candid with number (?)
        candidate = ZTFForcedPhotometryParser.parse(
            fp, candid=alert["candid"], oid=oid, ra=alert["ra"], dec=alert["dec"]
        )
        candidate.update(
            {
                "candid": f'{candidate["candid"]}-{candidate["mjd"]}',
                "aid": aid,
                "has_stamp": False,  # ?
                "forced": True,
                "parent_candid": parent,
                "extra_fields": {
                    **alert["extra_fields"],
                    **candidate["extra_fields"],
                },
            }
        )
        candidate.pop("stamps")
        detections.append(candidate)

    return {
        "aid": alert["aid"],
        "detections": detections,
        "non_detections": non_detections,
    }
