STAMPS = {
    "type": "record",
    "name": "stamps",
    "fields": [
        {"name": "science", "type": ["null", "bytes"], "default": None},
        {"name": "template", "type": ["null", "bytes"], "default": None},
        {"name": "difference", "type": ["null", "bytes"], "default": None},
    ],
}

EXTRA_FIELDS = {
    "type": "map",
    "values": ["null", "int", "float", "string", "bytes"],
    "default": {},
}

SCHEMA = {
    "type": "record",
    "doc": "Multi stream alert of any telescope/survey",
    "name": "alerce.alert",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "oid", "type": "string"},
        {"name": "sid", "type": "string"},
        {"name": "tid", "type": "string"},
        {"name": "pid", "type": "long"},
        {"name": "candid", "type": "string"},
        {"name": "mjd", "type": "double"},
        {"name": "fid", "type": "string"},
        {"name": "ra", "type": "double"},
        {"name": "dec", "type": "double"},
        {"name": "mag", "type": "float"},
        {"name": "e_mag", "type": "float"},
        {"name": "isdiffpos", "type": "int"},
        {"name": "e_ra", "type": "float"},
        {"name": "e_dec", "type": "float"},
        {
            "name": "extra_fields",
            "type": EXTRA_FIELDS,
        },
        {"name": "stamps", "type": STAMPS},
    ],
}
