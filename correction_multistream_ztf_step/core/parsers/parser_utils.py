import pandas as pd
import numpy as np
import math

from db_plugins.db.sql.models import (
    ZtfDetection,
    ZtfForcedPhotometry,
    ForcedPhotometry,
    NonDetection,
    Detection,
)

GENERIC_FIELDS = [
    "tid",
    "sid",
    "oid",
    "pid",
    "mjd",
    "fid",
    "ra",
    "dec",
    "measurement_id",
    "isdiffpos",
    "parent_candid",
    "has_stamp",
    "magpsf",
    "sigmapsf",
    "mag",
    "e_mag",
]

CHANGE_VALUES = [
    "tid",
    "sid",
]

GENERIC_FIELDS_FP = [
    "tid",
    "sid",
    "oid",
    "pid",
    "mjd",
    "fid",
    "ra",
    "dec",
    "isdiffpos",
    "parent_candid",
    "has_stamp",
]

CHANGE_NAMES = {  # outside extrafields
    "magpsf": "mag",
    "sigmapsf": "e_mag",
}

CHANGE_NAMES_2 = {  # inside extrafields
    "sigmapsf_corr": "e_mag_corr",
    "sigmapsf_corr_ext": "e_mag_corr_ext",
    "magpsf_corr": "mag_corr",
}
ERRORS = {
    1: 0.065,
    2: 0.085,
    3: 0.01,
}

def _e_ra(dec, fid):
    try:
        return ERRORS[fid] / abs(math.cos(math.radians(dec)))
    except ZeroDivisionError:
        return float("nan")

def get_fid(fid_as_int: int):
    fid = {1: "g", 2: "r", 0: None, 12: "gr", 3: "i"}
    try:
        return fid[fid_as_int]
    except KeyError:
        return fid_as_int


def parse_output(result: dict):
    result["detections"] = pd.DataFrame(result["detections"]).groupby("oid")

    try:  # At least one non-detection
        result["non_detections"] = pd.DataFrame(result["non_detections"]).groupby("oid")
    except KeyError:  # to reproduce expected error for missing non-detections in loop
        result["non_detections"] = pd.DataFrame(columns=["oid"]).groupby("oid")
    output = []

    for oid, dets in result["detections"]:

        dets = dets.replace(
            {np.nan: None, pd.NA: None, -np.inf: None}
        )  # Avoid NaN in the final results or infinite
        for field in [
            "e_ra",
            "e_dec",
        ]:  # Replace the e_ra/e_dec converted to None back to float nan per avsc formatting
            dets[field] = dets[field].apply(lambda x: x if pd.notna(x) else float("nan"))
        unique_measurement_ids = result["measurement_ids"][oid]
        unique_measurement_ids_long = [int(id_str) for id_str in unique_measurement_ids]

        detections_result = dets.to_dict("records")

        # Force the detection' parent candid back to integer
        for detections in detections_result:
            detections["measurement_id"] = int(detections["measurement_id"])
            parent_candid = detections.get("parent_candid")
            if parent_candid is not None and pd.notna(parent_candid):
                detections["parent_candid"] = int(parent_candid)
            else:
                detections["parent_candid"] = None

        output_message = {
            "oid": oid,
            "measurement_id": unique_measurement_ids_long,
            "meanra": result["coords"][oid]["meanra"],
            "meandec": result["coords"][oid]["meandec"],
            "detections": detections_result,
        }

        try:
            output_message["non_detections"] = (
                result["non_detections"].get_group(oid).to_dict("records")
            )
        except KeyError:
            output_message["non_detections"] = []
        output.append(output_message)
    return output

def ddbb_to_dict(data: list, ztf: bool) -> list | tuple[list,list]:
    """
    The general idea is to take the data from ddbb and parser into list 
    of dictionaries. If the data comes direct from ztf, then we add the
    extra_fields field. We also make a change value to tid and sid
    """
    parsed_data = []
    if ztf:
        extra_fields_list = []

    for detection in data:
        detection: dict = detection[0].__dict__
        extra_fields = {}
        parsed_detections = {}
        for field, value in detection.items():
            if field.startswith("_"):
                continue
            elif not field in GENERIC_FIELDS and ztf:
                extra_fields[field] = value
            else:
                if field in CHANGE_VALUES:
                    parsed_detections[field] = 0
                else:
                    parsed_detections[field] = value
        parsed_data.append(parsed_detections)
        if ztf:
            extra_fields_list.append(extra_fields)

    if ztf:
        return parsed_data, extra_fields_list
    else:
        return parsed_data
    
def model_instance(model_class, data: dict) -> dict:
    return model_class(**data)

def dicts_through_model(data: list[dict], model_class) -> list[dict]:
    """ Generic function to apply to every dict the appropiate model"""
    parsed_list = []

    for detection in data:
        parsed = model_instance(model_class, detection)
        parsed_list.append(parsed)

    return parsed_list

def join_ztf(parsed_dets, parsed_ztf_list, parsed_ztf_detections, extra_fields_list, det):
    """ 
    
    Join the ztf data with the non ztf data. Also add the extra_fields.
    If the data is detections, then we change some names.
    
    """
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

        if det:
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


    return parsed_ztf_list

def calc_ra_dec(dict_parsed):

    for d in dict_parsed:
        e_ra = _e_ra(d["dec"], d["band"])
        e_dec = ERRORS[d["band"]]

        d["e_ra"] = e_ra
        d["e_dec"] = e_dec

        del d["_sa_instance_state"]

    return dict_parsed