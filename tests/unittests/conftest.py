from pytest import fixture


def get_binary(filename: str):
    with open(f"tests/data/{filename}.fits.gz", "rb") as f:
        data = f.read()
        return data


@fixture
def alerts():
    alert = {
        "oid": "oid",
        "tid": "tid",
        "pid": 1,
        "candid": "123",
        "mjd": 1,
        "fid": 1,
        "ra": 1,
        "dec": 1,
        "rb": 1,
        "rbversion": "a",
        "mag": 1,
        "e_mag": 1,
        "rfid": 1,
        "isdiffpos": 1,
        "e_ra": 1,
        "e_dec": 1,
        "extra_fields": {},
        "aid": "aid",
        "stamps": {
            "science": get_binary("science"),
            "template": None,
            "difference": get_binary("difference"),
        },
    }
    return [alert]


@fixture
def ztf_alerts():
    alert = {
        "oid": "oid",
        "tid": "tid",
        "pid": 1,
        "candid": "123",
        "mjd": 2459994.9891782 - 2400000.5,
        "fid": 1,
        "ra": 208.1339812,
        "dec": 39.9475878,
        "rb": 1,
        "rbversion": "a",
        "mag": 1,
        "e_mag": 1,
        "rfid": 1,
        "isdiffpos": 0,
        "e_ra": 1,
        "e_dec": 1,
        "extra_fields": {
            "ndethist": 17,
            "ncovhist": 2512,
            "jdstarthist": 2458246.7887731,
            "jdendhist": 2459994.9891782,
            "ssdistnr": -999,
            "sgscore1": 0.97491,
            "distpsnr1": 0.215099
        },
        "aid": "aid",
        "stamps": {
            "science": get_binary("ztf_science"),
            "template": get_binary("ztf_template"),
            "difference": get_binary("ztf_difference"),
        },
    }
    return [alert]
