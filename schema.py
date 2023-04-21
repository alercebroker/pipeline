DETECTIONS = {
    "type": "array",
    "default": [],
    "items": {
        "name": "detections_record",
        "type": "record",
        "fields": [
            {"name": "candid", "type": ["long", "string"]},
            {"name": "tid", "type": "string"},
            {"name": "aid", "type": "string"},
            {"name": "oid", "type": "string"},
            {"name": "mjd", "type": "double"},
            {"name": "sid", "type": "string"},
            {"name": "fid", "type": "string"},
            {"name": "pid", "type": "long"},
            {"name": "ra", "type": "double"},
            {"name": "e_ra", "type": "float"},
            {"name": "dec", "type": "double"},
            {"name": "e_dec", "type": "float"},
            {"name": "mag", "type": "float"},
            {"name": "e_mag", "type": "float"},
            {"name": "mag_corr", "type": ["float", "null"]},
            {"name": "e_mag_corr", "type": ["float", "null"]},
            {"name": "e_mag_corr_ext", "type": ["float", "null"]},
            {"name": "isdiffpos", "type": "int"},
            {"name": "corrected", "type": "boolean"},
            {"name": "dubious", "type": "boolean"},
            {"name": "has_stamp", "type": "boolean"},
            {"name": "stellar", "type": "boolean"},
            {
                "name": "extra_fields",
                "type": {
                    "default": {},
                    "type": "map",
                    "values": [
                        "null",
                        "int",
                        "float",
                        "string",
                        "bytes",
                        "boolean",
                    ],
                },
            },
        ],
    },
}

NON_DETECTIONS = {
    "type": "array",
    "default": [],
    "items": {
        "name": "non_detections_record",
        "type": "record",
        "fields": [
            {"name": "aid", "type": "string"},
            {"name": "tid", "type": "string"},
            {"name": "sid", "type": "string"},
            {"name": "oid", "type": "string"},
            {"name": "mjd", "type": "double"},
            {"name": "fid", "type": "int"},
            {"name": "diffmaglim", "type": "float"},
        ],
    },
}

XMATCH = {
    "type": "map",
    "values": {"type": "map", "values": ["string", "float", "null", "int"]},
}

SCHEMA = {
    "doc": "Multi stream light curve with xmatch",
    "name": "alerce.light_curve_xmatched",
    "type": "record",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "meanra", "type": "float"},
        {"name": "meandec", "type": "float"},
        {"name": "detections", "type": DETECTIONS},
        {"name": "non_detections", "type": NON_DETECTIONS},
        {"name": "xmatches", "type": [XMATCH, "null"]},
    ],
}
