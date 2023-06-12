def ztf_extra_fields(**kwargs):
    extra_fields = {
        "magnr": 10.0,
        "sigmagnr": 1.0,
        "distnr": 1.0,
        "distpsnr1": 1.0,
        "sgscore1": 0.5,
        "chinr": 1,
        "sharpnr": 0.0,
        "unused": None,
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
        "aid": "AID1",
        "fid": "g",
        "mjd": 1.0,
        "has_stamp": True,
        "forced": False,
        "extra_fields": ztf_extra_fields(),
    }
    alert.update(kwargs)
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
        "aid": "AID1",
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
        "aid": "AID1",
    }
    alert.update(kwargs)
    return alert
