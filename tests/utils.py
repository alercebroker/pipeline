def ztf_extra_fields(**kwargs):
    extra_fields = {
        "magnr": 10.,
        "sigmagnr": 1.,
        "distnr": 1.,
        "distpsnr1": 1.,
        "sgscore1": .5,
        "chinr": 1,
        "sharpnr": 0.,
        "unused": None,
    }
    extra_fields.update(kwargs)
    return extra_fields


def ztf_alert(**kwargs):
    alert = {
        "candid": "candid",
        "tid": "ZTF",
        "mag": 10.,
        "e_mag": 10.,
        "ra": 1,
        "e_ra": 1,
        "dec": 1,
        "e_dec": 1,
        "isdiffpos": 1,
        "aid": "AID1",
        "fid": 1,
        "mjd": 1.,
        "has_stamp": True,
        "extra_fields": ztf_extra_fields()
    }
    alert.update(kwargs)
    return alert


def atlas_alert(**kwargs):
    alert = {
        "candid": "candid",
        "tid": "ATLAS-0",
        "mag": 10.,
        "e_mag": 10.,
        "ra": 1,
        "e_ra": 1,
        "dec": 1,
        "e_dec": 1,
        "isdiffpos": 1,
        "aid": "AID1",
        "fid": 1,
        "mjd": 1.,
        "has_stamp": True,
        "extra_fields": {"dummy": None}
    }
    alert.update(kwargs)
    return alert


def non_detection(**kwargs):
    alert = {
        "aid": "AID1",
    }
    alert.update(kwargs)
    return alert
