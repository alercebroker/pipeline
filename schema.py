METADATA = {
    "name": "metadata",
    "type": "record",
    "fields": [
        {
            "name": "ps1",
            "type": {
                "name": "ps1",
                "type": "record",
                "fields": [
                    {"name": "simag1", "type": ["float", "null"]},
                    {"name": "objectidps3", "type": ["double", "null"]},
                    {"name": "objectidps1", "type": ["double", "null"]},
                    {"name": "unique1", "type": ["boolean", "null"]},
                    {"name": "unique2", "type": ["boolean", "null"]},
                    {"name": "szmag2", "type": ["double", "null"]},
                    {"name": "srmag3", "type": ["float", "null"]},
                    {"name": "sgscore1", "type": ["float", "null"]},
                    {"name": "szmag3", "type": ["float", "null"]},
                    {"name": "srmag1", "type": ["float", "null"]},
                    {"name": "sgmag1", "type": ["float", "null"]},
                    {"name": "szmag1", "type": ["float", "null"]},
                    {"name": "distpsnr1", "type": ["float", "null"]},
                    {"name": "sgscore2", "type": ["float", "null"]},
                    {"name": "candid", "type": ["long", "null"]},
                    {"name": "simag3", "type": ["float", "null"]},
                    {"name": "objectidps2", "type": ["double", "null"]},
                    {"name": "srmag2", "type": ["float", "null"]},
                    {"name": "unique3", "type": ["boolean", "null"]},
                    {"name": "sgmag3", "type": ["float", "null"]},
                    {"name": "sgmag2", "type": ["double", "null"]},
                    {"name": "simag2", "type": ["float", "null"]},
                    {"name": "distpsnr2", "type": ["float", "null"]},
                    {"name": "distpsnr3", "type": ["float", "null"]},
                    {"name": "nmtchps", "type": ["int", "null"]},
                    {"name": "sgscore3", "type": ["float", "null"]},
                ],
            },
        },
        {
            "name": "ss",
            "type": {
                "name": "ss",
                "type": "record",
                "fields": [
                    {"name": "ssdistnr", "type": ["double", "null"]},
                    {"name": "ssmagnr", "type": ["double", "null"]},
                    {"name": "ssnamenr", "type": ["string", "null"]},
                ],
            },
        },
        {
            "name": "gaia",
            "type": {
                "name": "gaia",
                "type": "record",
                "fields": [
                    {"name": "maggaiabright", "type": ["float", "null"]},
                    {"name": "neargaiabright", "type": ["float", "null"]},
                    {"name": "unique1", "type": "boolean"},
                    {"name": "neargaia", "type": ["float", "null"]},
                    {"name": "maggaia", "type": ["float", "null"]},
                ],
            },
        },
    ],
}

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
            {"name": "corrected", "type": "boolean"},
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

SCHEMA = {
    "doc": "Multi stream light curve",
    "name": "alerce.light_curve",
    "type": "record",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "candid", "type": ["string", "long"]},
        {"name": "meanra", "type": "float"},
        {"name": "meandec", "type": "float"},
        {"name": "ndet", "type": "int"},
        {"name": "detections", "type": DETECTIONS},
        {"name": "non_detections", "type": NON_DETECTIONS},
        {"name": "metadata", "type": [METADATA, "null"]},
    ],
}
