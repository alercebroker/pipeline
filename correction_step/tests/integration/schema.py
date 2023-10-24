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
        {"name": "candid", "type": ["long", "string"]},
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
        {"name": "new", "type": "boolean"},
        {"name": "parent_candid", "type": ["long", "string", "null"]},
        {
            "name": "extra_fields",
            "type": {"default": {}, "type": "map", "values": ["null", "int", "float", "string", "bytes", "boolean"]},
        },
    ],
    "default": [],
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
    "default": [],
}

SCHEMA = {
    "type": "record",
    "doc": "Lightcurve schema with detections and non-detections",
    "name": "lightcurve",
    "fields": [
        {"name": "aid", "type": "string"},
        {
            "name": "detections",
            "type": {"type": "array", "items": DETECTION},
        },
        {
            "name": "non_detections",
            "type": {"type": "array", "items": NON_DETECTION},
        },
    ],
}
