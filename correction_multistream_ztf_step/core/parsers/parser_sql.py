from db_plugins.db.sql.models import (
    ZtfDetection,
    ZtfForcedPhotometry,
    ForcedPhotometry,
    NonDetection,
    Detection,
)
import math

# CHANGE_VALUES is used to pass certain values to 0. In this case, tid and sid.
# CHANGE_NAMES (and _2) are suposed to parser some names to the correct ones.

# Constant imports
from core.parsers.parser_utils import (
    GENERIC_FIELDS,
    CHANGE_VALUES,
    CHANGE_NAMES,
    CHANGE_NAMES_2,
    ERRORS,
)

# Function imports
from core.parsers.parser_utils import (
    _e_ra,
    ddbb_to_dict,
    dicts_through_model
)






def parse_sql_detection(ztf_models: list, models: list, *, oids) -> list:
    """
    Here we join the ztf_detections and detections in one dict for every oid. Also we add the field extra_fields to the final result
    and we change some names helped by the CHANGE_NAMES dict in the top of the code.

    """

    parsed_ztf_detections, extra_fields_list = ddbb_to_dict(ztf_models, ztf=True)
    parsed_dets = ddbb_to_dict(models, ztf=False)

    # Here we form the parsed_ztf object for every dict in parsed_ztf_detections.
    parsed_ztf_list = dicts_through_model(parsed_ztf_detections, ZtfDetection)


    
    # Here we join detections and ztf_detections in one. Also we hardcode forced and new as False because this is from the DDBB
    # and is not a new object. On the other land as we are in detections, this is not forced so forced is false. Also we join the extra_fields field
    # and change some names in this field.
    for detections in parsed_dets:
        for key, value in detections.items():
            if not key in parsed_ztf_detections[parsed_dets.index(detections)].keys():
                setattr(parsed_ztf_list[parsed_dets.index(detections)], key, value)

        setattr(parsed_ztf_list[parsed_dets.index(detections)], "forced", False)
        setattr(parsed_ztf_list[parsed_dets.index(detections)], "new", False)
        setattr(
            parsed_ztf_list[parsed_dets.index(detections)],
            "extra_fields",
            extra_fields_list[parsed_dets.index(detections)],
        )
        for name in CHANGE_NAMES.keys():
            parsed_ztf_list[parsed_dets.index(detections)].__dict__[CHANGE_NAMES[name]] = (
                parsed_ztf_list[parsed_dets.index(detections)].__dict__[name]
            )
            del parsed_ztf_list[parsed_dets.index(detections)].__dict__[name]

        for name in CHANGE_NAMES_2.keys():
            parsed_ztf_list[parsed_dets.index(detections)].__dict__["extra_fields"][
                CHANGE_NAMES_2[name]
            ] = parsed_ztf_list[parsed_dets.index(detections)].__dict__["extra_fields"][name]
            del parsed_ztf_list[parsed_dets.index(detections)].__dict__["extra_fields"][name]

    dict_parsed = list(map(vars, parsed_ztf_list))

    # Here we calculate e_ra and e_dec and join into every final dict. These variables are used to calculate meanra and meandec in correction.
    for d in dict_parsed:
        e_ra = _e_ra(d["dec"], d["band"])
        e_dec = ERRORS[d["band"]]

        d["e_ra"] = e_ra
        d["e_dec"] = e_dec

        d["sid"] = 0
        d["tid"] = 0

    for d in dict_parsed:
        del d["_sa_instance_state"]
        
    return dict_parsed


def parse_sql_non_detection(ztf_models: list, *, oids) -> list:
    non_dets = []

    for d in ztf_models:
        parsed_non_det = {}
        for field, value in d[0].__dict__.items():
            if field.startswith("_"):
                continue
            else:
                if field in CHANGE_VALUES:
                    parsed_non_det[field] = 0
                else:
                    parsed_non_det[field] = value
        non_dets.append(parsed_non_det)

    non_dets_parsed = []
    for non_det in non_dets:
        non_detection = NonDetection(**non_det)
        non_dets_parsed.append(non_detection)

    dict_parsed = list(map(vars, non_dets_parsed))
    for d in dict_parsed:
        del d["_sa_instance_state"]
        d["sid"] = 0
        d["tid"] = 0

    return dict_parsed


def parse_sql_forced_photometry(ztf_models: list, models: list, *, oids) -> list:
    """
    Here we join the ztf_fp and fp in one dict for every oid. Also we add the field extra_fields to the final result.
    """
    parsed_ztf_detections, extra_fields_list = ddbb_to_dict(ztf_models, ztf=True)

    parsed_dets = ddbb_to_dict(ztf_models, ztf=False)


    parsed_ztf_list = dicts_through_model(parsed_ztf_detections, ZtfForcedPhotometry)


    # Here we join ztf_fp and fp and put forced to true and new to false.
    for detections in parsed_dets:
        for key, value in detections.items():
            if not key in parsed_ztf_detections[parsed_dets.index(detections)].keys():
                setattr(parsed_ztf_list[parsed_dets.index(detections)], key, value)

        setattr(parsed_ztf_list[parsed_dets.index(detections)], "forced", True)
        setattr(parsed_ztf_list[parsed_dets.index(detections)], "new", False)
        setattr(
            parsed_ztf_list[parsed_dets.index(detections)],
            "extra_fields",
            extra_fields_list[parsed_dets.index(detections)],
        )

    dict_parsed = list(map(vars, parsed_ztf_list))

    for d in dict_parsed:
        del d["_sa_instance_state"]
        d["sid"] = 0
        d["tid"] = 0
        del d["pid"]  # Temporal fix to avoid error (Not all pids have value in the new DDBB)
        d["pid"] = 0

    return dict_parsed
