def ztf_extra_fields(**kwargs):
    extra_fields = {
        "magnr": 10.,
        "sigmagnr": 1.,
        "distnr": 1.,
        "unused": None,
    }
    extra_fields.update(kwargs)
    return extra_fields


def generate_alert(**kwargs):
    alert = {
        "candid": "candid",
        "mag": 10.,
        "e_mag": 10.,
        "isdiffpos": 1,
        "aid": "AID1",
        "fid": 1,
        "mjd": 1.,
        "extra_fields": {}
    }
    alert.update(kwargs)
    return alert
