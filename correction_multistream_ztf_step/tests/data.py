objects = [
  {
    "oid": "1234567890",
    "tid": 1,
    "sid": 2,
    "meanra": 5.8,
    "meandec": -11.1,
    "sigmara": 0.066,
    "sigmadec": 0.065,
    "firstmjd": 60684.1,
    "lastmjd": 60684.1,
    "deltamjd": 0.0,
    "n_det": 1,
    "n_forced": 0,
    "n_non_det": 1480,
    "corrected": False,
    "stellar": False
  },
  {
    "oid": "non_duped_oid",
    "tid": 1,
    "sid": 2,
    "meanra": 5.8,
    "meandec": -11.1,
    "sigmara": 0.066,
    "sigmadec": 0.065,
    "firstmjd": 60684.1,
    "lastmjd": 60694.1,
    "deltamjd": 0.0,
    "n_det": 1,
    "n_forced": 0,
    "n_non_det": 1480,
    "corrected": False,
    "stellar": False
  },{
    "oid": "duped_oid",
    "tid": 1,
    "sid": 2,
    "meanra": 5.8,
    "meandec": -11.1,
    "sigmara": 0.056,
    "sigmadec": 0.065,
    "firstmjd": 60684.1,
    "lastmjd": 60684.1,
    "deltamjd": 0.0,
    "n_det": 1,
    "n_forced": 0,
    "n_non_det": 1480,
    "corrected": False,
    "stellar": False
  }
]

detections = [
    {
        'oid':'1234567890',
        'measurement_id': 987654321,
        'mjd': 0.4124512,
        'ra': 152.3524,
        'dec': -89.51242,
        'band': 1
    }, 
    {
        'oid':'non_duped_oid',
        'measurement_id': 1,
        'mjd': 0.4124512,
        'ra': 152.3524,
        'dec': -89.51242,
        'band': 2
    },
    {
        'oid':'duped_oid',
        'measurement_id': 0,
        'mjd': 0.5124513,
        'ra': 152.3524,
        'dec': -89.51242,
        'band': 1
    },
    {
        'oid':'duped_oid',
        'measurement_id': 2,
        'mjd': 0.4124512,
        'ra': 152.3524,
        'dec': -89.51242,
        'band': 1
    }

]


ztf_detections = [
  {
    "oid": "1234567890",
    "measurement_id": 987654321,
    "pid": 6677889,
    "diffmaglim": 19.8,
    "isdiffpos": -1,
    "nid": 2002,
    "magpsf": 17.5,
    "sigmapsf": 0.12,
    "magap": 17.4,
    "sigmagap": 0.18,
    "distnr": 2.5,
    "rb": 0.9,
    "rbversion": "v1.2",
    "drb": 0.95,
    "drbversion": "v1.3",
    "magapbig": 17.3,
    "sigmagapbig": 0.22,
    "rband": 2,
    "magpsf_corr": 17.45,
    "sigmapsf_corr": 0.11,
    "sigmapsf_corr_ext": 0.10,
    "corrected": True,
    "dubious": True,
    "parent_candid": 9876543211,
    "has_stamp": False,
    "step_id_corr": 2
  },
  {
    "oid": "non_duped_oid",
    "measurement_id": 1,
    "pid": 6677889,
    "diffmaglim": 19.8,
    "isdiffpos": -1,
    "nid": 2002,
    "magpsf": 17.5,
    "sigmapsf": 0.12,
    "magap": 17.4,
    "sigmagap": 0.18,
    "distnr": 2.5,
    "rb": 0.9,
    "rbversion": "v1.2",
    "drb": 0.95,
    "drbversion": "v1.3",
    "magapbig": 17.3,
    "sigmagapbig": 0.22,
    "rband": 2,
    "magpsf_corr": 17.45,
    "sigmapsf_corr": 0.11,
    "sigmapsf_corr_ext": 0.10,
    "corrected": True,
    "dubious": True,
    "parent_candid": 9876543211,
    "has_stamp": False,
    "step_id_corr": 2
  },
  {
    "oid": "duped_oid",
    "measurement_id": 0,
    "pid": 6677889,
    "diffmaglim": 19.8,
    "isdiffpos": -1,
    "nid": 2002,
    "magpsf": 17.5,
    "sigmapsf": 0.12,
    "magap": 17.4,
    "sigmagap": 0.18,
    "distnr": 2.5,
    "rb": 0.9,
    "rbversion": "v1.2",
    "drb": 0.95,
    "drbversion": "v1.3",
    "magapbig": 17.3,
    "sigmagapbig": 0.22,
    "rband": 2,
    "magpsf_corr": 17.45,
    "sigmapsf_corr": 0.11,
    "sigmapsf_corr_ext": 0.10,
    "corrected": False,
    "dubious": True,
    "parent_candid": 9876543211,
    "has_stamp": False,
    "step_id_corr": 2
  },
  {
    "oid": "duped_oid",
    "measurement_id": 2,
    "pid": 6677889,
    "diffmaglim": 19.8,
    "isdiffpos": -1,
    "nid": 2002,
    "magpsf": 17.5,
    "sigmapsf": 0.12,
    "magap": 17.4,
    "sigmagap": 0.18,
    "distnr": 2.5,
    "rb": 0.9,
    "rbversion": "v1.2",
    "drb": 0.95,
    "drbversion": "v1.3",
    "magapbig": 17.3,
    "sigmagapbig": 0.22,
    "rband": 2,
    "magpsf_corr": 17.45,
    "sigmapsf_corr": 0.11,
    "sigmapsf_corr_ext": 0.10,
    "corrected": False,
    "dubious": True,
    "parent_candid": 9876543211,
    "has_stamp": False,
    "step_id_corr": 2
  }
]

non_detection = [
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60655.155775499996,
        "diffmaglim": 17.6075
    },
    {
        "oid": "1234567890",
        "band": 2,
        "mjd": 60658.11518519977,
        "diffmaglim": 18.2779
    },
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60658.19327550009,
        "diffmaglim": 19.0953
    },
    {
        "oid": "1234567890",
        "band": 2,
        "mjd": 60660.149386600126,
        "diffmaglim": 19.4483
    },
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60660.2327430998,
        "diffmaglim": 18.7138
    },
    {
        "oid": "1234567890",
        "band": 2,
        "mjd": 60663.106655099895,
        "diffmaglim": 19.9582
    },
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60663.22942129988,
        "diffmaglim": 18.8907
    },
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60664.11018519988,
        "diffmaglim": 20.1121
    },
    {
        "oid": "1234567890",
        "band": 2,
        "mjd": 60664.14053239999,
        "diffmaglim": 20.2564
    },
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60666.120231499895,
        "diffmaglim": 19.9874
    },
    {
        "oid": "1234567890",
        "band": 2,
        "mjd": 60666.12964119995,
        "diffmaglim": 20.0196
    },
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60668.110752299894,
        "diffmaglim": 18.8554
    },
    {
        "oid": "1234567890",
        "band": 2,
        "mjd": 60668.180081000086,
        "diffmaglim": 17.0589
    },
    {
        "oid": "1234567890",
        "band": 2,
        "mjd": 60672.10907409992,
        "diffmaglim": 19.6796
    },
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60672.1296064998,
        "diffmaglim": 19.6858
    },
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60674.09282410005,
        "diffmaglim": 20.1046
    },
    {
        "oid": "1234567890",
        "band": 2,
        "mjd": 60674.155729200225,
        "diffmaglim": 19.8949
    },
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60677.194907399826,
        "diffmaglim": 19.6374
    },
    {
        "oid": "1234567890",
        "band": 1,
        "mjd": 60679.10571759986,
        "diffmaglim": 19.8679
    },
    {
        "oid": "1234567890",
        "band": 2,
        "mjd": 60679.185705999844,
        "diffmaglim": 19.6428
    },
    {
        "oid": "non_duped_oid",
        "band": 2,
        "mjd": 60679.185705999844,
        "diffmaglim": 19.6428
    },
    {
        "oid": "non_duped_oid",
        "band": 1,
        "mjd": 60679.285705999844,
        "diffmaglim": 20.4587
    },
    {
        "oid": "duped_oid",
        "band": 1,
        "mjd": 60726.28568289988,
        "diffmaglim": 20.403099060058594
    },
    {
        "oid": "duped_oid",
        "band": 2,
        "mjd": 60728.33390050009,
        "diffmaglim": 19.596900939941406
    }

]

ztf_forced_photometry = [{
    "oid": "1234567890",
    "measurement_id": 123456789987654321,
    "mag": 18.45,
    "e_mag": 0.12,
    "mag_corr": 18.37,
    "e_mag_corr": 0.09,
    "e_mag_corr_ext": 0.11,
    "isdiffpos": 1,
    "corrected": True,
    "dubious": False,
    "parent_candid": "ZTF19abcdefg",
    "has_stamp": True,
    "field": 456,
    "rcid": 3,
    "rband": 2,
    "sciinpseeing": 1.23,
    "scibckgnd": 25.67,
    "scisigpix": 8.97,
    "magzpsci": 26.54,
    "magzpsciunc": 0.03,
    "magzpscirms": 0.02,
    "clrcoeff": 0.15,
    "clrcounc": 0.02,
    "exptime": 30.0,
    "adpctdif1": 0.35,
    "adpctdif2": 0.42,
    "diffmagli": 19.23,
    "programid": 1,
    "procstatus": "SUCCESS",
    "distnr": 0.76,
    "ranr": 182.45678,
    "decnr": 23.12345,
    "magnr": 17.89,
    "sigmagnr": 0.05,
    "chinr": 1.12,
    "sharpnr": 0.23
},{
    "oid": "non_duped_oid",
    "measurement_id": 10,
    "mag": 18.45,
    "e_mag": 0.12,
    "mag_corr": 18.37,
    "e_mag_corr": 0.09,
    "e_mag_corr_ext": 0.11,
    "isdiffpos": 1,
    "corrected": True,
    "dubious": False,
    "parent_candid": "ZTF19abcdefg",
    "has_stamp": True,
    "field": 456,
    "rcid": 3,
    "rband": 2,
    "sciinpseeing": 1.23,
    "scibckgnd": 25.67,
    "scisigpix": 8.97,
    "magzpsci": 26.54,
    "magzpsciunc": 0.03,
    "magzpscirms": 0.02,
    "clrcoeff": 0.15,
    "clrcounc": 0.02,
    "exptime": 30.0,
    "adpctdif1": 0.35,
    "adpctdif2": 0.42,
    "diffmagli": 19.23,
    "programid": 1,
    "procstatus": "SUCCESS",
    "distnr": 0.76,
    "ranr": 182.45678,
    "decnr": 23.12345,
    "magnr": 17.89,
    "sigmagnr": 0.05,
    "chinr": 1.12,
    "sharpnr": 0.23
},{
    "oid": "duped_oid",
    "measurement_id": 2,
    "mag": 10.45,
    "e_mag": 0.12,
    "mag_corr": 18.37,
    "e_mag_corr": 0.09,
    "e_mag_corr_ext": 0.11,
    "isdiffpos": 1,
    "corrected": True,
    "dubious": False,
    "parent_candid": "ZTF19abcdefg",
    "has_stamp": True,
    "field": 456,
    "rcid": 3,
    "rband": 2,
    "sciinpseeing": 1.23,
    "scibckgnd": 25.67,
    "scisigpix": 8.97,
    "magzpsci": 26.54,
    "magzpsciunc": 0.03,
    "magzpscirms": 0.02,
    "clrcoeff": 0.15,
    "clrcounc": 0.02,
    "exptime": 30.0,
    "adpctdif1": 0.35,
    "adpctdif2": 0.42,
    "diffmagli": 19.23,
    "programid": 1,
    "procstatus": "SUCCESS",
    "distnr": 0.76,
    "ranr": 182.45678,
    "decnr": 23.12345,
    "magnr": 17.89,
    "sigmagnr": 0.05,
    "chinr": 1.12,
    "sharpnr": 0.23
},{
    "oid": "duped_oid",
    "measurement_id": 1,
    "mag": 10.45,
    "e_mag": 0.12,
    "mag_corr": 18.37,
    "e_mag_corr": 0.09,
    "e_mag_corr_ext": 0.11,
    "isdiffpos": 1,
    "corrected": True,
    "dubious": False,
    "parent_candid": "ZTF19abcdefg",
    "has_stamp": True,
    "field": 456,
    "rcid": 3,
    "rband": 2,
    "sciinpseeing": 1.23,
    "scibckgnd": 25.67,
    "scisigpix": 8.97,
    "magzpsci": 26.54,
    "magzpsciunc": 0.03,
    "magzpscirms": 0.02,
    "clrcoeff": 0.15,
    "clrcounc": 0.02,
    "exptime": 30.0,
    "adpctdif1": 0.35,
    "adpctdif2": 0.42,
    "diffmagli": 19.23,
    "programid": 1,
    "procstatus": "SUCCESS",
    "distnr": 0.76,
    "ranr": 182.45678,
    "decnr": 23.12345,
    "magnr": 17.89,
    "sigmagnr": 0.05,
    "chinr": 1.12,
    "sharpnr": 0.23
}

]

forced_photometry = [{
    "oid": "1234567890",
    "measurement_id":123456789987654321,
    "mjd": 60671.155729200225,
    'ra': 150.3524,
    'dec': -19.51242,
    "band": 1
},{
    "oid": "non_duped_oid",
    "measurement_id":10,
    "mjd": 60671.155729200225,
    'ra': 150.3524,
    'dec': -19.51242,
    "band": 1
},{
    "oid": "duped_oid",
    "measurement_id":2,
    "mjd": 60571.155729200225,
    'ra': 154.3524,
    'dec': 21.51242,
    "band": 1
},{
    "oid": "duped_oid",
    "measurement_id":1,
    "mjd": 60571.155729200225,
    'ra': 154.3524,
    'dec': 21.51242,
    "band": 1
}

]