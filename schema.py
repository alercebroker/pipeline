DETECTIONS = {
    "type": "array",
    "items": {
        "name": "detections_record",
        "type": "record",
        "fields": [
            {"name": "aid", "type": "string"},
            {"name": "oid", "type": "string"},
            {"name": "sid", "type": "string"},
            {"name": "candid", "type": "string"},
            {"name": "mjd", "type": "double"},
            {"name": "fid", "type": "int"},
            {"name": "ra", "type": "double"},
            {"name": "dec", "type": "double"},
            {"name": "rb", "type": ["float", "null"]},
            {"name": "mag", "type": "float"},
            {"name": "sigmag", "type": "float"},
            {"name": "aimage", "type": ["float", "null"]},
            {"name": "bimage", "type": ["float", "null"]},
            {"name": "xpos", "type": ["float", "null"]},
            {"name": "ypos", "type": ["float", "null"]},
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
            {"name": "sid", "type": "string"},
            {"name": "oid", "type": "string"},
            {"name": "mjd", "type": "double"},
            {"name": "diffmaglim", "type": "float"},
            {"name": "fid", "type": "int"},
        ],
    },
}
# ["aid", "sid", "oid", "mjd", "diffmaglim", "fid"]
SCHEMA = {
    "doc": "Light curve",
    "name": "light_curve",
    "type": "record",
    "fields": [
        {"name": "oid", "type": "string"},
        {
            "name": "candid",
            "type": [
                "string",
                {
                    "type": "array",
                    "items": "string",
                    "default": []
                }
            ]
        },
        {"name": "detections", "type": DETECTIONS},
        {"name": "non_detections", "type": NON_DETECTIONS},
    ],
}
