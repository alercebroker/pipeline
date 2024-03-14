from db_plugins.db.mongo.models import (
    Detection as MongoDetection,
    NonDetection as MongoNonDetection,
    ForcedPhotometry as MongoForcedPhotometry,
)
from .parser_utils import get_fid


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
        "corrected",
        "dubious",
        "candid",
        "parent_candid",
        "has_stamp",
    }
    FIELDS_TO_REMOVE = [
        "stellar",
        "e_mag_corr",
        "corrected",
        "mag_corr",
        "e_mag_corr_ext",
        "dubious",
        "magpsf",
        "sigmapsf",
        "magpsf_corr",
        "sigmapsf_corr",
        "sigmapsf_corr_ext",
    ]

    parsed_result = []
    for det in ztf_models:
        det: dict = det[0].__dict__
        extra_fields = {}
        parsed_det = {}
        for field, value in det.items():
            if field.startswith("_"):
                continue
            if field not in GENERIC_FIELDS:
                extra_fields[field] = value
            if field == "fid":
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
            e_ra=-999,
            e_dec=-999,
        )
        parsed.pop("_id", None)

        for field in FIELDS_TO_REMOVE:
            parsed.pop(field, None)
            parsed["extra_fields"].pop(field, None)

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
        fp["candid"] = fp.pop("_id", None)
        fp["extra_fields"] = {
            k: v
            for k, v in fp["extra_fields"].items()
            if not k.startswith("_")
        }
        # remove problematic fields
        FIELDS_TO_REMOVE = [
            "stellar",
            "e_mag_corr",
            "corrected",
            "mag_corr",
            "e_mag_corr_ext",
            "dubious",
        ]
        for field in FIELDS_TO_REMOVE:
            fp.pop(field, None)

        return fp

    parsed = [
        {
            **MongoForcedPhotometry(
                **forced[0].__dict__,
                aid=forced[0].__dict__.get("aid"),
                sid="ZTF",
                tid="ZTF",
                candid=forced[0].__dict__["oid"]
                + str(forced[0].__dict__["pid"]),
            ),
            "new": False,
            "forced": True,
        }
        for forced in ztf_models
    ]

    return list(map(format_as_detection, parsed))
