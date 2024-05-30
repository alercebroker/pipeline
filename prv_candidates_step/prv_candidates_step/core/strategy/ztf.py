import copy
import numpy as np
import pickle

from survey_parser_plugins.core import SurveyParser
from survey_parser_plugins.core.generic import GenericNonDetection
from survey_parser_plugins.core.mapper import Mapper
from survey_parser_plugins.parsers import ZTFParser

_ZERO_MAG = 100.0


def prv_non_detections_mapper() -> dict:
    mapping = copy.deepcopy(ZTFParser._mapping)
    preserve = ["oid", "sid", "tid", "fid", "mjd", "aid"]

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
        return message["candid"] is not None

    @classmethod
    def parse_message(cls, candidate: dict, alert) -> dict:
        acopy = copy.deepcopy(alert)
        candcopy = copy.deepcopy(candidate)
        candcopy["objectId"] = acopy["oid"]
        generic = {name: mapper(candcopy) for name, mapper in cls._mapping.items()}

        stamps = cls._extract_stamps(candcopy)
        extra_fields = {
            k: v
            for k, v in candcopy.items()
            if k not in cls._exclude_from_extra_fields()
        }
        model = cls._Model(**generic, stamps=stamps, extra_fields=extra_fields)
        model = model.to_dict()
        model.pop("stamps", None)
        model.update(
            {
                "aid": acopy["aid"],
                "has_stamp": False,
                "forced": False,
                "parent_candid": acopy["candid"],
            }
        )
        return model


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

        if np.isclose(data["forcediffimflux"], -99999):
            mag = _ZERO_MAG
            e_mag = _ZERO_MAG

        if np.isclose(data["forcediffimfluxunc"], -99999):
            e_mag = _ZERO_MAG

        return mag, e_mag

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(ZTFForcedPhotometryParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, message: dict) -> bool:
        return True

    @classmethod
    def parse_message(cls, forced_photometry: dict, alert: dict) -> dict:
        fpcopy = copy.deepcopy(forced_photometry)
        acopy = copy.deepcopy(alert)
        fpcopy["candid"] = acopy["oid"] + str(forced_photometry["pid"])
        fpcopy["magpsf"], fpcopy["sigmapsf"] = cls.__calculate_mag(fpcopy)
        fpcopy["objectId"] = acopy["oid"]

        fpcopy["ra"] = acopy["ra"]
        fpcopy["dec"] = acopy["dec"]
        generic = {name: mapper(fpcopy) for name, mapper in cls._mapping.items()}

        stamps = cls._extract_stamps(fpcopy)
        extra_fields = {
            k: v
            for k, v in forced_photometry.items()
            if k not in cls._exclude_from_extra_fields()
        }
        model = cls._Model(**generic, stamps=stamps, extra_fields=extra_fields)
        model = model.to_dict()

        model.update(
            {
                "aid": acopy["aid"],
                "has_stamp": False,
                "forced": True,
                "parent_candid": acopy["candid"],
            }
        )
        model.pop("stamps")
        return model


class ZTFNonDetectionsParser(SurveyParser):
    _mapping = prv_non_detections_mapper()
    _Model = GenericNonDetection

    @classmethod
    def _extract_stamps(cls, message: dict) -> dict:
        return super(ZTFNonDetectionsParser, cls)._extract_stamps(message)

    @classmethod
    def can_parse(cls, candidate: dict) -> bool:
        return candidate["candid"] is None

    @classmethod
    def parse_message(cls, candidate: dict, alert: dict) -> dict:
        candcopy = copy.deepcopy(candidate)
        acopy = copy.deepcopy(alert)
        candcopy["objectId"] = acopy["oid"]
        generic = {name: mapper(candcopy) for name, mapper in cls._mapping.items()}

        stamps = cls._extract_stamps(candcopy)
        extra_fields = {
            k: v
            for k, v in candcopy.items()
            if k not in cls._exclude_from_extra_fields()
        }
        model = cls._Model(**generic, stamps=stamps, extra_fields=extra_fields)
        model = model.to_dict()
        model.update({"aid": acopy["aid"]})
        model.pop("stamps", None)
        model.pop("extra_fields", None)
        return model


def extract_detections_and_non_detections(alert: dict) -> dict:
    prv_candidates = alert["extra_fields"].pop("prv_candidates")
    prv_candidates = pickle.loads(prv_candidates) if prv_candidates else []

    forced_photometries = alert["extra_fields"].pop("fp_hists", [])
    forced_photometries = (
        pickle.loads(forced_photometries) if forced_photometries else []
    )

    # filter forced photometry by value
    # potencial bugs from ztf
    def filter_fp(fp):
        forcediffimflux_bad_values = [None, 0]
        forcediffimfluxunc_bad_values = [None, 0]
        # if the value is not close to -99999 return true
        good_fp_flux = fp[
            "forcediffimflux"
        ] not in forcediffimflux_bad_values and not np.isclose(
            fp["forcediffimflux"], -99999
        )
        good_fp_fluxunc = fp[
            "forcediffimfluxunc"
        ] not in forcediffimflux_bad_values and not np.isclose(
            fp["forcediffimfluxunc"], -99999
        )
        return good_fp_flux and good_fp_fluxunc

    # use the filter funcion to remove bad fp
    forced_photometries = list(filter(filter_fp, forced_photometries))

    acopy = copy.deepcopy(alert)
    detections = [acopy]
    non_detections = []
    detections = detections + [
        ZTFPreviousDetectionsParser.parse_message(candidate, acopy)
        for candidate in prv_candidates
        if ZTFPreviousDetectionsParser.can_parse(candidate)
    ]
    non_detections = [
        ZTFNonDetectionsParser.parse_message(candidate, acopy)
        for candidate in prv_candidates
        if ZTFNonDetectionsParser.can_parse(candidate)
    ]
    acopy["extra_fields"]["parent_candid"] = None
    detections = detections + [
        ZTFForcedPhotometryParser.parse_message(fp, acopy)
        for fp in forced_photometries
        if ZTFForcedPhotometryParser.can_parse(fp)
    ]

    return {
        "oid": acopy["oid"],
        "detections": detections,
        "non_detections": non_detections,
    }
