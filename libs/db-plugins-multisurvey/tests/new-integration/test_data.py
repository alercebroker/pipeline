"""
Archivo con los datos de prueba para los tests de modelos SQLAlchemy.
Contiene diccionarios con datos para los modelos Object, ZtfObject, Detection, ZtfDetection y ForcedPhotometry.
"""
# Datos para el modelo Object
OBJECT_DATA = {
    "filter": [
        {
            "oid": 12345680,
            "tid": 1,
            "sid": 1,
            "meanra": 293.26922,
            "meandec": 74.38752,
            "sigmara": 0.000326,
            "sigmade": 0.000064,
            "firstmjd": 60058.47743,
            "lastmjd": 60207.21820,
            "deltamjd": 148.74077,
            "n_det": 44,
            "n_forced": 12,
            "n_non_det": 7,
            "corrected": False,
            "stellar": True
        },
        {
            "oid": 12345681,
            "tid": 2,
            "sid": 1,
            "meanra": 290.54321,
            "meandec": 71.98765,
            "sigmara": 0.000412,
            "sigmade": 0.000089,
            "firstmjd": 60060.32145,
            "lastmjd": 60210.54321,
            "deltamjd": 150.22176,
            "n_det": 38,
            "n_forced": 15,
            "n_non_det": 5,
            "corrected": True,
            "stellar": False
        },
        {
            "oid": 12345682,
            "tid": 1,
            "sid": 2,
            "meanra": 292.87654,
            "meandec": 73.12345,
            "sigmara": 0.000287,
            "sigmade": 0.000053,
            "firstmjd": 60055.76543,
            "lastmjd": 60205.98765,
            "deltamjd": 150.22222,
            "n_det": 45,
            "n_forced": 8,
            "n_non_det": 3,
            "corrected": True,
            "stellar": True
        }
    ]
}

# Datos para el modelo ZtfObject
ZTF_OBJECT_DATA = {
    "filter": [
        {
            "oid": 12345680,
            "g_r_max": 0.43,
            "g_r_max_corr": 1.2,
            "g_r_mean": 3.2,
            "g_r_mean_corr": 0.01
        },
        {
            "oid": 12345681,
            "g_r_max": 0.55,
            "g_r_max_corr": 1.4,
            "g_r_mean": 2.8,
            "g_r_mean_corr": 0.02
        },
        {
            "oid": 12345682,
            "g_r_max": 0.65,
            "g_r_max_corr": 1.6,
            "g_r_mean": 3.4,
            "g_r_mean_corr": 0.03
        },
        {
            "oid": 12345683,
            "g_r_max": 0.35,
            "g_r_max_corr": 1.1,
            "g_r_mean": 2.5,
            "g_r_mean_corr": 0.015
        }
    ]
}

# Datos para el modelo Detection
DETECTION_DATA = {
    "filter": [
        {
            "oid": 12345680,
            "measurement_id": 1001,
            "mjd": 60080.5432,
            "ra": 293.26945,
            "dec": 74.38762,
            "band": 1
        },
        {
            "oid": 12345680,
            "measurement_id": 1002,
            "mjd": 60085.6543,
            "ra": 293.26930,
            "dec": 74.38758,
            "band": 2
        },
        {
            "oid": 12345680,
            "measurement_id": 1003,
            "mjd": 60090.7654,
            "ra": 293.26918,
            "dec": 74.38750,
            "band": 1
        }
    ]
}

# Datos para el modelo ZtfDetection
ZTF_DETECTION_DATA = {
    "filter": [
        {
            "oid": 12345680,
            "measurement_id": 987654321,
            "pid": 123456,
            "diffmaglim": 19.5,
            "isdiffpos": True,
            "nid": 1,
            "magpsf": 18.2,
            "sigmapsf": 0.1,
            "magap": 18.3,
            "sigmagap": 0.15,
            "distnr": 0.5,
            "rb": 0.8,
            "rbversion": "t2",
            "drb": 0.9,
            "drbversion": "t2",
            "magapbig": 18.0,
            "sigmagapbig": 0.2,
            "rband": 2,
            "magpsf_corr": 0,
            "sigmapsf_corr": 0,
            "sigmapsf_corr_ext": 0,
            "corrected": False,
            "dubious": False,
            "parent_candid": 12345,
            "has_stamp": True,
            "step_id_corr": 0
        },
        {
            "oid": 12345680,
            "measurement_id": 987654322,
            "pid": 123457,
            "diffmaglim": 19.6,
            "isdiffpos": True,
            "nid": 2,
            "magpsf": 17.8,
            "sigmapsf": 0.12,
            "magap": 17.9,
            "sigmagap": 0.16,
            "distnr": 0.6,
            "rb": 0.85,
            "rbversion": "t2",
            "drb": 0.91,
            "drbversion": "t2",
            "magapbig": 17.7,
            "sigmagapbig": 0.21,
            "rband": 2,
            "magpsf_corr": 0,
            "sigmapsf_corr": 0,
            "sigmapsf_corr_ext": 0,
            "corrected": False,
            "dubious": False,
            "parent_candid": 12346,
            "has_stamp": True,
            "step_id_corr": 0
        },
        {
            "oid": 12345681,
            "measurement_id": 987654323,
            "pid": 123458,
            "diffmaglim": 19.7,
            "isdiffpos": False,
            "nid": 3,
            "magpsf": 18.5,
            "sigmapsf": 0.13,
            "magap": 18.6,
            "sigmagap": 0.17,
            "distnr": 0.7,
            "rb": 0.86,
            "rbversion": "t2",
            "drb": 0.92,
            "drbversion": "t2",
            "magapbig": 18.3,
            "sigmagapbig": 0.22,
            "rband": 1,
            "magpsf_corr": 0,
            "sigmapsf_corr": 0,
            "sigmapsf_corr_ext": 0,
            "corrected": False,
            "dubious": True,
            "parent_candid": 12347,
            "has_stamp": False,
            "step_id_corr": 0
        },
        {
            "oid": 12345682,
            "measurement_id": 987654324,
            "pid": 123459,
            "diffmaglim": 19.8,
            "isdiffpos": False,
            "nid": 4,
            "magpsf": 19.0,
            "sigmapsf": 0.15,
            "magap": 19.1,
            "sigmagap": 0.19,
            "distnr": 0.8,
            "rb": 0.75,
            "rbversion": "t2",
            "drb": 0.88,
            "drbversion": "t2",
            "magapbig": 18.9,
            "sigmagapbig": 0.25,
            "rband": 1,
            "magpsf_corr": 0,
            "sigmapsf_corr": 0,
            "sigmapsf_corr_ext": 0,
            "corrected": True,
            "dubious": False,
            "parent_candid": 12348,
            "has_stamp": True,
            "step_id_corr": 1
        }
    ]
}

# Datos para el modelo ForcedPhotometry
FORCED_PHOTOMETRY_DATA = {
    "filter": [
        {
            "oid": 12345680,
            "measurement_id": 987654321,
            "mjd": 58765.4321,
            "ra": 293.26945,
            "dec": 74.38762,
            "band": 1
        },
        {
            "oid": 12345680,
            "measurement_id": 987654322,
            "mjd": 58766.5432,
            "ra": 293.26930,
            "dec": 74.38758,
            "band": 2
        },
        {
            "oid": 12345681,
            "measurement_id": 987654323,
            "mjd": 58767.6543,
            "ra": 290.54335,
            "dec": 71.98775,
            "band": 1
        },
        {
            "oid": 12345682,
            "measurement_id": 987654324,
            "mjd": 58768.7654,
            "ra": 292.87670,
            "dec": 73.12360,
            "band": 3
        }
    ]
}

# Datos para el modelo ZtfForcedPhotometry
ZTF_FORCED_PHOTOMETRY_DATA = {
    "filter": [
        {
            "oid": 12345680,
            "measurement_id": 987654321,
            "mag": 18.5,
            "e_mag": 0.05,
            "mag_corr": 18.6,
            "e_mag_corr": 0.06,
            "e_mag_corr_ext": 0.07,
            "isdiffpos": 1,
            "corrected": False,
            "dubious": False,
            "parent_candid": "ZTF18abcdefgh",
            "has_stamp": True,
            "field": 567,
            "rcid": 42,
            "rband": 1,
            "sciinpseeing": 1.2,
            "scibckgnd": 25.5,
            "scisigpix": 10.2,
            "magzpsci": 26.5,
            "magzpsciunc": 0.02,
            "magzpscirms": 0.03,
            "clrcoeff": 0.05,
            "clrcounc": 0.01,
            "exptime": 30.0,
            "adpctdif1": 0.1,
            "adpctdif2": 0.2,
            "diffmagli": 19.5,
            "programid": 1,
            "procstatus": "Success",
            "distnr": 0.5,
            "ranr": 293.2690,
            "decnr": 74.3870,
            "magnr": 17.8,
            "sigmagnr": 0.03,
            "chinr": 1.2,
            "sharpnr": 0.5
        },
        {
            "oid": 12345680,
            "measurement_id": 987654322,
            "mag": 18.6,
            "e_mag": 0.06,
            "mag_corr": 18.7,
            "e_mag_corr": 0.07,
            "e_mag_corr_ext": 0.08,
            "isdiffpos": 1,
            "corrected": False,
            "dubious": False,
            "parent_candid": "ZTF18abcdefgh",
            "has_stamp": True,
            "field": 567,
            "rcid": 42,
            "rband": 2,
            "sciinpseeing": 1.3,
            "scibckgnd": 26.0,
            "scisigpix": 10.5,
            "magzpsci": 26.6,
            "magzpsciunc": 0.03,
            "magzpscirms": 0.04,
            "clrcoeff": 0.06,
            "clrcounc": 0.02,
            "exptime": 30.0,
            "adpctdif1": 0.15,
            "adpctdif2": 0.25,
            "diffmagli": 19.6,
            "programid": 1,
            "procstatus": "Success",
            "distnr": 0.6,
            "ranr": 293.2691,
            "decnr": 74.3871,
            "magnr": 17.9,
            "sigmagnr": 0.04,
            "chinr": 1.3,
            "sharpnr": 0.6
        },
        {
            "oid": 12345681,
            "measurement_id": 987654323,
            "mag": 19.2,
            "e_mag": 0.08,
            "mag_corr": 19.3,
            "e_mag_corr": 0.09,
            "e_mag_corr_ext": 0.1,
            "isdiffpos": 0,
            "corrected": True,
            "dubious": False,
            "parent_candid": "ZTF19abcdefgh",
            "has_stamp": True,
            "field": 568,
            "rcid": 43,
            "rband": 1,
            "sciinpseeing": 1.4,
            "scibckgnd": 25.8,
            "scisigpix": 10.8,
            "magzpsci": 26.7,
            "magzpsciunc": 0.04,
            "magzpscirms": 0.05,
            "clrcoeff": 0.07,
            "clrcounc": 0.02,
            "exptime": 30.0,
            "adpctdif1": 0.2,
            "adpctdif2": 0.3,
            "diffmagli": 19.7,
            "programid": 2,
            "procstatus": "Success",
            "distnr": 0.7,
            "ranr": 290.5430,
            "decnr": 71.9875,
            "magnr": 18.5,
            "sigmagnr": 0.05,
            "chinr": 1.4,
            "sharpnr": 0.7
        },
        {
            "oid": 12345682,
            "measurement_id": 987654324,
            "mag": 17.8,
            "e_mag": 0.04,
            "mag_corr": 17.9,
            "e_mag_corr": 0.05,
            "e_mag_corr_ext": 0.06,
            "isdiffpos": 0,
            "corrected": True,
            "dubious": True,
            "parent_candid": "ZTF20abcdefgh",
            "has_stamp": False,
            "field": 569,
            "rcid": 44,
            "rband": 3,
            "sciinpseeing": 1.5,
            "scibckgnd": 26.2,
            "scisigpix": 11.0,
            "magzpsci": 26.8,
            "magzpsciunc": 0.05,
            "magzpscirms": 0.06,
            "clrcoeff": 0.08,
            "clrcounc": 0.03,
            "exptime": 30.0,
            "adpctdif1": 0.25,
            "adpctdif2": 0.35,
            "diffmagli": 19.8,
            "programid": 3,
            "procstatus": "Partial",
            "distnr": 0.8,
            "ranr": 292.8764,
            "decnr": 73.1233,
            "magnr": 16.9,
            "sigmagnr": 0.02,
            "chinr": 1.5,
            "sharpnr": 0.8
        }
    ]
}

# Datos para el modelo NonDetection
NON_DETECTION_DATA = {
    "filter": [
        {
            "oid": 12345680,
            "band": 1,
            "mjd": 60030.12345,
            "diffmaglim": 19.5
        },
        {
            "oid": 12345680,
            "band": 2,
            "mjd": 60035.23456,
            "diffmaglim": 19.8
        },
        {
            "oid": 12345681,
            "band": 1,
            "mjd": 60040.34567,
            "diffmaglim": 20.1
        },
        {
            "oid": 12345682,
            "band": 3,
            "mjd": 60045.45678,
            "diffmaglim": 20.5
        }
    ]
}