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
