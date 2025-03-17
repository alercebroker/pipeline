from db_plugins.db.mongo.models import (
    Detection as MongoDetection,
    NonDetection as MongoNonDetection,
    ForcedPhotometry as MongoForcedPhotometry,
)
from .parser_utils import get_fid
from survey_parser_plugins.parsers.ZTFParser import _e_ra
from survey_parser_plugins.parsers.ZTFParser import ERRORS as dec_errors


def parse_sql_detection(ztf_models: list, *, oids) -> list:
    GENERIC_FIELDS = {
        "tid",
        "sid",
        "aid",
        "oid",
        "pid",
        "mjd",
        "fid",
        "ra",
        "dec",
        "isdiffpos",
        "candid",
        "parent_candid",
        "has_stamp",
        "corrected",  # for dumb a reason: MongoDetection needs it
        "dubious",  # idem as above
    }

    # db has old names for corrected mags in detection table
    # update names to schema convention
    corrected_mag_name_map = {
        "magpsf_corr": "mag_corr",
        "sigmapsf_corr": "e_mag_corr",
        "sigmapsf_corr_ext": "e_mag_corr_ext",
    }

    parsed_result = []
    for det in ztf_models:
        det: dict = det[0].__dict__
        extra_fields = {}
        parsed_det = {}
        for field, value in det.items():
            if field.startswith("_"):
                continue
            elif field in corrected_mag_name_map.keys():
                extra_fields[corrected_mag_name_map[field]] = value
            elif field not in GENERIC_FIELDS:
                extra_fields[field] = value
            elif field == "fid":
                parsed_det[field] = get_fid(value)
            else:
                parsed_det[field] = value

        parsed = MongoDetection(
            **parsed_det,
            aid=det.get("aid", None),
            sid="ZTF",
            tid="ZTF",
            mag=det["magpsf"],
            e_mag=det["sigmapsf"],
            mag_corr=det["magpsf_corr"],
            e_mag_corr=det["sigmapsf_corr"],
            e_mag_corr_ext=det["sigmapsf_corr_ext"],
            extra_fields=extra_fields,
            e_ra=_e_ra(det["dec"], det["fid"]),
            e_dec=dec_errors[det["fid"]],
        )
        parsed.pop("_id", None)

        # Corrected mags belong to extra fields in the lightcurve step schema
        # In the correction step schema they become detection fields
        parsed.pop("mag_corr")
        parsed.pop("e_mag_corr")
        parsed.pop("e_mag_corr_ext")

        # corrected and dubious belong to extra fields in the lightcurve step schema
        # In the correction step schema they become detection fields
        parsed["extra_fields"]["corrected"] = parsed.pop("corrected")
        parsed["extra_fields"]["dubious"] = parsed.pop("dubious")

        parsed["extra_fields"].pop("magpsf")
        parsed["extra_fields"].pop("sigmapsf")

        assert {"corrected", "dubious"}.issubset(parsed["extra_fields"].keys())
        parsed_result.append({**parsed, "forced": False, "new": False})

    return parsed_result


def parse_sql_non_detection(ztf_models: list, *, oids) -> list:
    non_dets = []
    for non_det in ztf_models:
        non_det = non_det[0].__dict__
        mongo_non_detection = MongoNonDetection(
            _id="jej",
            tid="ZTF",
            sid="ZTF",
            aid=non_det.get("aid"),
            oid=non_det["oid"],
            mjd=non_det["mjd"],
            fid=get_fid(non_det["fid"]),
            diffmaglim=non_det.get("diffmaglim", None),
        )
        mongo_non_detection.pop("_id", None)
        mongo_non_detection.pop("extra_fields", None)
        non_dets.append(mongo_non_detection)
    return non_dets


def parse_sql_forced_photometry(ztf_models: list, *, oids) -> list:
    def format_as_detection(fp):
        fp["fid"] = get_fid(fp["fid"])
        fp["e_ra"] = 0
        fp["e_dec"] = 0
        fp["candid"] = fp["oid"] + str(fp["pid"])
        fp["extra_fields"] = {
            k: v
            for k, v in fp["extra_fields"].items()
            if not k.startswith("_")
        }

        # remove problematic fields
        FIELDS_TO_REMOVE = [
            "stellar",
            "corrected",
            # "mag_corr",
            # "e_mag_corr",
            # "e_mag_corr_ext",
            "dubious",
        ]
        for field in FIELDS_TO_REMOVE:
            fp.pop(field, None)

        return fp

    parsed = []
    for forced in ztf_models:
        parsed_fp = {
            **MongoForcedPhotometry(
                **forced[0].__dict__,
                aid=forced[0].__dict__.get("aid"),
                sid="ZTF",
                tid="ZTF",
            ),
            "new": False,
            "forced": True,
        }

        # corrected magnitude fields must go in extra fields for feature step output
        parsed_fp["extra_fields"]["mag_corr"] = parsed_fp["mag_corr"]
        parsed_fp["extra_fields"]["e_mag_corr"] = parsed_fp["e_mag_corr"]
        parsed_fp["extra_fields"]["e_mag_corr_ext"] = parsed_fp[
            "e_mag_corr_ext"
        ]

        del parsed_fp["mag_corr"]
        del parsed_fp["e_mag_corr"]
        del parsed_fp["e_mag_corr_ext"]

        parsed.append(parsed_fp)

    return list(map(format_as_detection, parsed))
