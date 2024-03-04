import pickle
from random import random, choice


def ztf_extra_fields(**kwargs):
    extra_fields = {
        "magnr": 10.0,
        "sigmagnr": 1.0,
        # make some detections non correctible
        "distnr": random() + 1 if random() < 0.5 else random(),
        "distpsnr1": 1.0,
        "sgscore1": 0.5,
        "chinr": 1,
        "sharpnr": 0.0,
        "unused": None,
    }
    extra_fields.update(kwargs)
    return extra_fields


def lsst_extra_fields(**kwargs):
    extra_fields = {
        "field": "value",
        "prvDiaForcedSources": b"bainari",
        "prvDiaSources": b"bainari2",
        "diaObject": pickle.dumps("bainari2"),
    }
    extra_fields.update(kwargs)
    return extra_fields


def atlas_extra_fields(**kwargs):
    extra_fields = {
        "field": "value",
    }
    extra_fields.update(kwargs)
    return extra_fields


def ztf_alert(**kwargs):
    alert = {
        "candid": "candid",
        "sid": "ZTF",
        "tid": "ZTF",
        "mag": 10.0,
        "e_mag": 10.0,
        "ra": 1,
        "e_ra": 1,
        "dec": 1,
        "e_dec": 1,
        "isdiffpos": 1,
        "oid": "OID1",
        "fid": "g",
        "mjd": 1.0,
        "has_stamp": True,
        "forced": choice([True, False]),
        "extra_fields": kwargs["extra_fields"]
        if kwargs.get("extra_fields", None)
        else ztf_extra_fields(),
    }
    alert.update(kwargs)
    return alert


def elasticc_extra_fields():
    extra_fields = {
        "diaObject": pickle.dumps("bainari_diaObject"),
        "prvDiaSources": pickle.dumps("bainari_prvDiaSoruces"),
        "prvDiaForcedSources": pickle.dumps("bainari_prvForced"),
    }
    return extra_fields


def elasticc_alert(**kwargs):
    alert = ztf_alert(**kwargs)
    alert["tid"] = "LSST"
    alert["sid"] = "LSST"
    alert["extra_fields"] = elasticc_extra_fields()
    return alert


def atlas_alert(**kwargs):
    alert = {
        "candid": "candid",
        "sid": "ATLAS",
        "tid": "ATLAS-0",
        "mag": 10.0,
        "e_mag": 10.0,
        "ra": 1,
        "e_ra": 1,
        "dec": 1,
        "e_dec": 1,
        "isdiffpos": 1,
        "oid": "OID1",
        "fid": "c",
        "mjd": 1.0,
        "has_stamp": True,
        "forced": False,
        "extra_fields": {"dummy": None},
    }
    alert.update(kwargs)
    return alert


def non_detection(**kwargs):
    alert = {
        "oid": "OID1",
    }
    alert.update(kwargs)
    return alert
