import pandas as pd

feature_columns_for_parse = [
    ("feat1", "g"),
    ("feat1", "r"),
    ("feat2", "gr"),
    ("feat3", ""),
    ("feat4", ""),
]
feature_multi_index_for_parse = pd.MultiIndex.from_tuples(
    feature_columns_for_parse
)

features_df_for_parse = pd.DataFrame(
    [[123, 456, 741, 963, None], [321, 654, 147, 369, 888]],
    index=["oid1", "oid2"],
    columns=feature_multi_index_for_parse,
)
features_df_for_parse.index.name = "oid"

messages_for_parsing = [
    {
        "oid": "oid1",
        "candid": "candid1",
        "meanra": 888,
        "meandec": 999,
        "detections": [],
        "non_detections": [],
        "xmatches": {},
    },
    {
        "oid": "oid2",
        "candid": "candid2",
        "meanra": 444,
        "meandec": 555,
        "detections": [],
        "non_detections": [],
        "xmatches": {},
    },
]

feature_columns_for_execute = [
    ("Amplitude", "g"),
    ("Amplitude", "r"),
    ("Multiband_period", "gr"),
    ("feat3", ""),
    ("rb", ""),
]
feature_multi_index_for_execute = pd.MultiIndex.from_tuples(
    feature_columns_for_execute
)

features_df_for_execute = pd.DataFrame(
    [[123, 456, 741, 963, None], [321, 654, 147, 369, 888]],
    index=["oid1", "oid2"],
    columns=feature_multi_index_for_execute,
)
features_df_for_execute.index.name = "oid"

messages_for_execute = [
    {
        "oid": "oid1",
        "candid": ["1_candid_aid_1", "2_candid_aid_1"],
        "meanra": 888,
        "meandec": 999,
        "detections": [
            {
                "candid": "1_candid_aid_1",
                "tid": "ztf",
                "aid": "aid1",
                "oid": "oid1",
                "mjd": 111,
                "sid": "sid-aid1",
                "fid": "1",
                "pid": 222,
                "ra": 333,
                "e_ra": 444,
                "dec": 555,
                "e_dec": 666,
                "mag": 777,
                "e_mag": 888,
                "mag_corr": None,
                "e_mag_corr": None,
                "e_mag_corr_ext": None,
                "isdiffpos": 999,
                "corrected": False,
                "dubious": False,
                "has_stamp": True,
                "stellar": False,
                "extra_fields": {},
            },
            {
                "candid": "2_candid_aid_1",
                "tid": "ztf",
                "aid": "aid1",
                "oid": "oid1",
                "mjd": 111,
                "sid": "sid-aid1",
                "fid": "1",
                "pid": 222,
                "ra": 333,
                "e_ra": 444,
                "dec": 555,
                "e_dec": 666,
                "mag": 777,
                "e_mag": 888,
                "mag_corr": None,
                "e_mag_corr": None,
                "e_mag_corr_ext": None,
                "isdiffpos": 999,
                "corrected": False,
                "dubious": False,
                "has_stamp": True,
                "stellar": False,
                "extra_fields": {},
            },
        ],
        "non_detections": [
            {
                "aid": "aid1",
                "tid": "ztf",
                "sid": "sid_aid1",
                "oid": "oid1",
                "mjd": 999.888,
                "fid": "1",
                "diffmaglim": 123.123,
            },
            {
                "aid": "aid1",
                "tid": "ztf",
                "sid": "sid_aid1",
                "oid": "oid1",
                "mjd": 888,
                "fid": "1",
                "diffmaglim": 999,
            },
        ],
        "xmatches": {
            "W1mag": 123,
            "W2mag": 456,
            "W3mag": 789,
        },
    },
    {
        "oid": "oid2",
        "candid": ["1_candid_aid_2"],
        "meanra": 444,
        "meandec": 555,
        "detections": [
            {
                "candid": "1_candid_aid_2",
                "tid": "ztf",
                "aid": "aid2",
                "oid": "oid2",
                "mjd": 111,
                "sid": "sid-aid1",
                "fid": "1",
                "pid": 222,
                "ra": 333,
                "e_ra": 444,
                "dec": 555,
                "e_dec": 666,
                "mag": 777,
                "e_mag": 888,
                "mag_corr": None,
                "e_mag_corr": None,
                "e_mag_corr_ext": None,
                "isdiffpos": 999,
                "corrected": False,
                "dubious": False,
                "has_stamp": True,
                "stellar": False,
                "extra_fields": {},
            }
        ],
        "non_detections": [
            {
                "aid": "aid2",
                "tid": "ztf",
                "sid": "sid_aid2",
                "oid": "oid2",
                "mjd": 888,
                "fid": "1",
                "diffmaglim": 999,
            }
        ],
        "xmatches": {
            "W1mag": 123,
            "W2mag": 456,
            "W3mag": 789,
        },
    },
]
