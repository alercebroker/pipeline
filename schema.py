# TODO define more generic xmatch schema
XMATCH = {}
# TODO define more generic metadata schema
METADATA = {}

DETECTIONS = {
    "type": "array",
    "items": {
        "name": "detections_record",
        "type": "record",
        "fields": [
            {"name": "oid", "type": "string"},
            {"name": "candid", "type": "long"},
            {"name": "mjd", "type": "float"},
            {"name": "fid", "type": "int"},
            {"name": "pid", "type": "float"},
            {"name": "diffmaglim", "type": "float"},
            {"name": "isdiffpos", "type": "int"},
            {"name": "nid", "type": "int"},
            {"name": "ra", "type": "float"},
            {"name": "dec", "type": "float"},
            {"name": "magpsf", "type": "float"},
            {"name": "sigmapsf", "type": "float"},
            {"name": "magap", "type": "float"},
            {"name": "sigmagap", "type": "float"},
            {"name": "distnr", "type": "float"},
            {"name": "rb", "type": "float"},
            {"name": "rbversion", "type": "string"},
            {"name": "drb", "type": ["float", "null"]},
            {"name": "drbversion", "type": ["string", "null"]},
            {"name": "magapbig", "type": "float"},
            {"name": "sigmagapbig", "type": "float"},
            {"name": "rfid", "type": ["int", "null"]},
            {"name": "magpsf_corr", "type": "float"},
            {"name": "sigmapsf_corr", "type": "float"},
            {"name": "sigmapsf_corr_ext", "type": "float"},
            {"name": "corrected", "type": "boolean"},
            {"name": "dubious", "type": "boolean"},
            {"name": "parent_candid", "type": ["long", "null"]},
            {"name": "has_stamp", "type": "boolean"},
            {"name": "step_id_corr", "type": "string"},
        ],
    },
}

NON_DETECTIONS = {
    "type": "array",
    "items": {
        "name": "non_detections_record",
        "type": "record",
        "fields": [
            {"name": "oid", "type": "string"},
            {"name": "mjd", "type": "float"},
            {"name": "diffmaglim", "type": "float"},
            {"name": "fid", "type": "int"},
        ],
    },
}

METADATA_2 = {
    "name": "metadata",
    "type": "record",
    "fields": [
        {
            "name": "ps1",
            "type": {
                "name": "ps1",
                "type": "record",
                "fields": [
                    {"name": "simag1", "type": "float"},
                    {"name": "objectidps3", "type": "double"},
                    {"name": "objectidps1", "type": "double"},
                    {"name": "unique1", "type": "boolean"},
                    {"name": "unique2", "type": "boolean"},
                    {"name": "szmag2", "type": "double"},
                    {"name": "srmag3", "type": "float"},
                    {"name": "sgscore1", "type": "float"},
                    {"name": "szmag3", "type": "float"},
                    {"name": "srmag1", "type": "float"},
                    {"name": "sgmag1", "type": "float"},
                    {"name": "szmag1", "type": "float"},
                    {"name": "distpsnr1", "type": "float"},
                    {"name": "sgscore2", "type": "float"},
                    {"name": "candid", "type": "long"},
                    {"name": "simag3", "type": "float"},
                    {"name": "objectidps2", "type": "double"},
                    {"name": "srmag2", "type": "float"},
                    {"name": "unique3", "type": "boolean"},
                    {"name": "sgmag3", "type": "float"},
                    {"name": "sgmag2", "type": "double"},
                    {"name": "simag2", "type": "float"},
                    {"name": "distpsnr2", "type": "float"},
                    {"name": "distpsnr3", "type": "float"},
                    {"name": "nmtchps", "type": "int"},
                    {"name": "sgscore3", "type": "float"},
                ],
            },
        },
        {
            "name": "ss",
            "type": {
                "name": "ss",
                "type": "record",
                "fields": [
                    {"name": "ssdistnr", "type": "double"},
                    {"name": "ssmagnr", "type": "double"},
                    {"name": "ssnamenr", "type": "string"},
                ],
            },
        },
        {
            "name": "reference",
            "type": {
                "name": "reference",
                "type": "record",
                "fields": [
                    {"name": "magnr", "type": "float"},
                    {"name": "ranr", "type": "float"},
                    {"name": "field", "type": "int"},
                    {"name": "chinr", "type": "float"},
                    {"name": "mjdstartref", "type": "float"},
                    {"name": "mjdendref", "type": "float"},
                    {"name": "decnr", "type": "float"},
                    {"name": "sharpnr", "type": "float"},
                    {"name": "candid", "type": "long"},
                    {"name": "nframesref", "type": "int"},
                    {"name": "rcid", "type": "int"},
                    {"name": "rfid", "type": "int"},
                    {"name": "fid", "type": "int"},
                    {"name": "sigmagnr", "type": "float"},
                ],
            },
        },
        {
            "name": "gaia",
            "type": {
                "name": "gaia",
                "type": "record",
                "fields": [
                    {"name": "maggaiabright", "type": "float"},
                    {"name": "neargaiabright", "type": "float"},
                    {"name": "unique", "type": "boolean"},
                    {"name": "neargaia", "type": "float"},
                    {"name": "maggaia", "type": "float"},
                ],
            },
        },
    ],
}

SCHEMA = {
    "doc": "Light curve",
    "name": "light_curve",
    "type": "record",
    "fields": [
        {"name": "oid", "type": "string"},
        {"name": "candid", "type": "long"},
        {"name": "detections", "type": DETECTIONS},
        {"name": "non_detections", "type": NON_DETECTIONS,},
        {"name": "xmatches", "type": ["string", "null"]},
        {"name": "fid", "type": "int"},
        {"name": "metadata", "type": METADATA_2,},
    ],
}
