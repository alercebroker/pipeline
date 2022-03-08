DETECTIONS = {
    "type": "array",
    "items": {
        "name": "detections_record",
        "type": "record",
        "fields": [
            # {"name": "aid", "type": "string"},
            {"name": "oid", "type": "string"},
            {"name": "tid", "type": "string"},
            {"name": "candid", "type": ["string", "int"]},
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
            {"name": "rbversion", "type": "string"},
            # {"name": "extra_fields", "type": {"type": "map", "values": "string"}},
        ],
    },
}

NON_DETECTIONS = {
    "type": "array",
    "items": {
        "name": "non_detections_record",
        "type": "record",
        "fields": [
            {"name": "aid", "type": "string"},
            {"name": "tid", "type": "string"},
            {"name": "oid", "type": "string"},
            {"name": "mjd", "type": "double"},
            {"name": "diffmaglim", "type": "float"},
            {"name": "fid", "type": "int"},
        ],
    },
}

SCHEMA = {
    "doc": "Multi stream light curve",
    "name": "alerce.light_curve",
    "type": "record",
    "fields": [
        {
            "name": "aid",
            "type": "string"},
        {
            "name": "candid",
            "type": "string"
        },
        {"name": "detections", "type": DETECTIONS},
        {"name": "non_detections", "type": NON_DETECTIONS},
    ],
}
