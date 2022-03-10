DETECTIONS = {
    "type": "array",
    "items": {
        "name": "detections_record",
        "type": "record",
        "fields": [
            {"name": "oid", "type": "string"},
            {"name": "tid", "type": "string"},
            {"name": "candid", "type": ["string", "long"]},
            {"name": "mjd", "type": "double"},
            {"name": "fid", "type": "int"},
            {"name": "ra", "type": "double"},
            {"name": "e_ra", "type": "double"},
            {"name": "dec", "type": "double"},
            {"name": "e_dec", "type": "double"},
            {"name": "mag", "type": "float"},
            {"name": "e_mag", "type": "float"},
            {"name": "isdiffpos", "type": "int"},
            {"name": "rb", "type": ["float", "null"]},
            {"name": "rbversion", "type": ["string", "null"]},
            {
                "name": "extra_fields",
                "type": {
                    "type": "map",
                    "values": ["string", "int", "null", "float", "boolean", "double"],
                },
            },
        ],
    },
}

NON_DETECTIONS = {
    "type": "array",
    "items": {
        "name": "non_detections_record",
        "type": "record",
        "fields": [
            {"name": "tid", "type": "string"},
            {"name": "oid", "type": "string"},
            {"name": "mjd", "type": "double"},
            {"name": "diffmaglim", "type": "float"},
            {"name": "fid", "type": "int"},
        ],
    },
}

XMATCH = {
    "type": "map",
    "values": {"type": "map", "values": ["string", "float", "null"]},
}

SCHEMA = {
    "doc": "Multi stream light curve with xmatch",
    "name": "alerce.light_curve_xmatched",
    "type": "record",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "candid", "type": ["string", "long"]},
        {"name": "meanra", "type": "float"},
        {"name": "meandec", "type": "float"},
        {"name": "ndet", "type": "int"},
        {"name": "detections", "type": DETECTIONS},
        {"name": "non_detections", "type": NON_DETECTIONS},
        {"name": "xmatches", "type": [XMATCH, "null"], "default": "null"},
    ],
}
