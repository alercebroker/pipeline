from db_plugins.db.sql.models import (
    ZtfDetection,
    ZtfForcedPhotometry,
    NonDetection,
)

# Function imports
from core.parsers.parser_utils import (
    ddbb_to_dict,
    dicts_through_model,
    join_ztf,
    calc_ra_dec,
)

def parse_sql_detection(ztf_models: list, models: list, *, oids) -> list:
    """
    Here we join the ztf_detections and detections in one dict for every oid. Also we add the field extra_fields to the final result
    and we change some names helped by the CHANGE_NAMES dict in the top of the code.

    """

    # We take the ddbb and pass to dict format with certain conditions.
    parsed_ztf_detections, extra_fields_list = ddbb_to_dict(ztf_models, ztf=True)
    parsed_dets = ddbb_to_dict(models, ztf=False)

    # Here we form the parsed_ztf object for every dict in parsed_ztf_detections.
    parsed_ztf_list = dicts_through_model(parsed_ztf_detections, ZtfDetection)
    
    # Here we join detections and ztf_detections in one. Also we hardcode forced and new as False because this is from the DDBB
    # and is not a new object. On the other land as we are in detections, this is not forced so forced is false. Also we join the extra_fields field
    # and change some names in this field.
    parsed_ztf_list = join_ztf(parsed_dets, parsed_ztf_list, parsed_ztf_detections, extra_fields_list, True)
    
    dict_parsed = list(map(vars, parsed_ztf_list))

    # Here we calculate e_ra and e_dec and join into every final dict. These variables are used to calculate meanra and meandec in correction.
    dict_parsed = calc_ra_dec(dict_parsed)

    return dict_parsed


def parse_sql_non_detection(models: list, *, oids) -> list:

    parsed_non_dets = ddbb_to_dict(models, ztf=False)
    non_dets_parsed = dicts_through_model(parsed_non_dets, NonDetection)
    dict_parsed = list(map(vars, non_dets_parsed))
    for d in dict_parsed:
        del d["_sa_instance_state"]
    print(dict_parsed)
    return dict_parsed


def parse_sql_forced_photometry(ztf_models: list, models: list, *, oids) -> list:
    """
    Here we join the ztf_fp and fp in one dict for every oid. Also we add the field extra_fields to the final result.
    
    The functions are the same as in detections.
    """
    parsed_ztf_detections, extra_fields_list = ddbb_to_dict(ztf_models, ztf=True)

    parsed_dets = ddbb_to_dict(ztf_models, ztf=False)

    parsed_ztf_list = dicts_through_model(parsed_ztf_detections, ZtfForcedPhotometry)


    # Here we join ztf_fp and fp and put forced to true and new to false.
    parsed_ztf_list = join_ztf(parsed_dets, parsed_ztf_list, parsed_ztf_detections, extra_fields_list, False)

    dict_parsed = list(map(vars, parsed_ztf_list))

    for d in dict_parsed:
        del d["_sa_instance_state"]

    return dict_parsed
