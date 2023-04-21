mock_lightcurve = [
    {
        "aid": "aid",
        "detections": [
            {
                "candid": "aaa",
                "tid": "ZTF",
                "oid": "oid",
                "aid": "aid",
                "mag": 0.1,
                "e_mag": 0.01,
                "extra_fields": {"field1": "f", "field2": "f2"},
            },
            {"candid": "aaa", "tid": "ATLAS-01a"},
        ],
        "magstats": [],
    }
]

mock_lightcurves_list = [
    {
        "aid": "aid1",
        "detections": [
            {
                "candid": "aaa",
                "tid": "ZTF",
                "oid": "oid1",
                "aid": "aid1",
                "mag": 0.1,
                "e_mag": 0.01,
                "extra_fields": {"field1": "f", "field2": "f2"},
            },
            {"candid": "aaa", "tid": "ATLAS-01a"},
        ],
        "non_detections": [],
    },
    {
        "aid": "aid2",
        "detections": [
            {
                "candid": "eee",
                "tid": "ZTF",
                "oid": "oid2",
                "aid": "aid2",
                "mag": 0.1,
                "e_mag": 0.01,
                "extra_fields": {"field1": "f", "field2": "f2"},
            },
            {"candid": "aaa2", "tid": "ATLAS-01a"},
        ],
        "non_detections": [
            {
                "tid": "ZTF",
                "oid": "oid2",
                "aid": "aid2",
                "mjd": 599000,
                "diffmaglim": 18,
                "fid": 1,
            }
        ],
    },
]

mock_xmatch_list = [
    {
        "ra_in": 123,
        "dec_in": 456,
        "col1": "col",
        "oid_in": "oid1",
        "aid_in": "aid1",
        "AllWISE": 999.888,
        "W1mag": 777.555,
        "W2mag": "J124131",
    },
    {
        "ra_in": 321,
        "dec_in": 654,
        "col1": "col2",
        "oid_in": "oid2",
        "aid_in": "aid2",
    },
]
