# previous detections schema


EXTRA_FIELDS = {
    "type": "map",
    "values": ["null", "int", "float", "string", "boolean", "bytes"],
    "default": {},
}

DETECTION = {
    "type": "record",
    "name": "detection",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "oid", "type": "string"},
        {"name": "sid", "type": "string"},
        {"name": "pid", "type": "long"},
        {"name": "tid", "type": "string"},
        {"name": "fid", "type": "string"},
        {"name": "candid", "type": "string"},
        {"name": "mjd", "type": "double"},
        {"name": "ra", "type": "double"},
        {"name": "e_ra", "type": "float"},
        {"name": "dec", "type": "double"},
        {"name": "e_dec", "type": "float"},
        {"name": "mag", "type": "float"},
        {"name": "e_mag", "type": "float"},
        {"name": "isdiffpos", "type": "int"},
        {"name": "has_stamp", "type": "boolean"},
        {"name": "forced", "type": "boolean"},
        {"name": "parent_candid", "type": ["string", "null"]},
        {"name": "new", "type": "boolean"},
        {
            "name": "extra_fields",
            "type": EXTRA_FIELDS,
        },
    ],
}

NON_DETECTION = {
    "type": "record",
    "name": "non_detection",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "oid", "type": "string"},
        {"name": "sid", "type": "string"},
        {"name": "tid", "type": "string"},
        {"name": "fid", "type": "string"},
        {"name": "mjd", "type": "double"},
        {"name": "diffmaglim", "type": "float"},
    ],
}

SCHEMA = {
    "type": "record",
    "doc": "Multi stream alert of any telescope/survey",
    "name": "prv_candidates",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "candid", "type": {"type": "array", "items": "string"}},
        {"name": "detections", "type": {"type": "array", "items": DETECTION}},
        {"name": "non_detections", "type": {"type": "array", "items": NON_DETECTION}},
    ],
}
