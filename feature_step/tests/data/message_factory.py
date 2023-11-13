import random

import pandas as pd

random.seed(8798, version=2)


def get_extra_fields(telescope: str):
    if telescope == "ATLAS":
        return {}
    elif telescope == "ZTF":
        return {
            "distnr": random.random(),
            "magnr": random.random(),
            "sigmagnr": random.random(),
            "pid": random.randint(1, 999999),
            "diffmaglim": random.uniform(15, 21),
            "nid": random.randint(1, 999999),
            "magpsf": random.random(),  #
            "sigmapsf": random.random(),  #
            "magap": random.random(),
            "sigmagap": random.random(),
            "magapbig": random.random(),
            "sigmagapbig": random.random(),
            "rb": random.uniform(0.55, 1),
            "sgscore1": random.uniform(0, 1),
        }


def generate_alert_atlas(num_messages: int, identifier: int) -> list[dict]:
    alerts = []
    for i in range(num_messages):
        alert = {
            "oid": f"ATLASoid{identifier}",
            "tid": "ATLAS",
            "candid": random.randint(1000000, 9000000),
            "mjd": random.uniform(59000, 60000),
            "fid": random.randint(1, 2),
            "ra": random.uniform(0, 360),
            "dec": random.uniform(-90, 90),
            "e_ra": random.random(),
            "e_dec": random.random(),
            "mag": random.uniform(15, 20),
            "e_mag": random.random(),
            "isdiffpos": random.choice([-1, 1]),
            "rb": random.random(),
            "rbversion": "v7379812",
            "aid": f"AL2X{random.randint(1000, 9990)}",
            "extra_fields": get_extra_fields("ATLAS"),
        }
        alerts.append(alert)
    return alerts


def generate_alert_ztf(
    aid: str, band: str, num_messages: int, identifier: int
) -> list[dict]:
    alerts = []
    for i in range(num_messages):
        alert = {
            "candid": str(random.randint(1000000, 9000000)),
            "oid": f"ZTFoid{identifier}",
            "aid": aid,
            "tid": "ZTF",
            "mjd": random.uniform(59000, 60000),
            "sid": "ZTF",
            "fid": band,
            "pid": random.randint(1000000, 9000000),
            "ra": random.uniform(-90, 90),
            "e_ra": random.uniform(-90, 90),
            "dec": random.uniform(-90, 90),
            "e_dec": random.uniform(-90, 90),
            "mag": random.uniform(15, 20),
            "e_mag": random.uniform(0, 1),
            "mag_corr": random.uniform(15, 20),
            "e_mag_corr": random.uniform(0, 1),
            "e_mag_corr_ext": random.uniform(0, 1),
            "isdiffpos": random.choice([-1, 1]),
            "corrected": random.choice([True, False]),
            "dubious": random.choice([True, False]),
            "has_stamp": random.choice([True, False]),
            "stellar": random.choice([True, False]),
            "new": random.choice([True, False]),
            "forced": random.choice([True, False]),
            "extra_fields": get_extra_fields("ZTF"),
        }
        alerts.append(alert)
    return alerts


def generate_non_det(aid: str, num: int, identifier: int) -> list[dict]:
    non_det = []
    for i in range(num):
        nd = {
            "aid": aid,
            "tid": "ztf",
            "oid": f"ZTFoid{identifier}",
            "sid": "ZTF",
            "mjd": random.uniform(59000, 60000),
            "fid": random.choice(["g", "r"]),
            "diffmaglim": random.uniform(15, 20),
        }
        non_det.append(nd)
    return non_det


def generate_input_batch(n: int) -> list[dict]:
    """
    Parameters
    ----------
    n: number of objects in the batch
    Returns a batch of generic message
    -------
    """
    batch = []
    for m in range(1, n + 1):
        aid = f"AL2X{str(m).zfill(5)}"
        meanra = random.uniform(0, 360)
        meandec = random.uniform(-90, 90)
        detections_g = generate_alert_ztf(aid, "g", random.randint(5, 10), m)
        detections_r = generate_alert_ztf(aid, "r", random.randint(5, 10), m)
        non_det = generate_non_det(aid, random.randint(1, 5), m)
        xmatch = get_fake_xmatch(aid, meanra, meandec)
        msg = {
            "aid": aid,
            "candid": [det["candid"] for det in detections_g + detections_r],
            "meanra": meanra,
            "meandec": meandec,
            "detections": detections_g + detections_r,
            "non_detections": non_det,
            "xmatches": xmatch,
        }
        batch.append(msg)
    random.sample(batch, len(batch))
    return batch


def get_default_object_values(identifier: int) -> dict:
    data = {
        "oid": f"ZTFoid{identifier}",
        "ndethist": 0.0,
        "ncovhist": 0.0,
        "mjdstarthist": 40000.0,
        "mjdendhist": 400000.0,
        "firstmjd": 400000.0,
        "lastmjd": 400000.0,
        "ndet": 1,
        "deltajd": 0,
        "meanra": 0.0,
        "meandec": 0.0,
        "step_id_corr": "test",
        "corrected": False,
        "stellar": False,
    }
    return data


def get_fake_xmatch(aid, meanra, meandec) -> pd.DataFrame:
    fake = {
        "allwise": {
            "angDist": round(random.uniform(0, 1), 6),
            "col1": random.randint(7, 10),
            "aid_in": aid,
            "ra_in": round(meanra, 6),
            "dec_in": round(meandec, 6),
            "AllWISE": f"J{random.randint(200000, 299999)}.32+240338.4",
            "RAJ2000": round(meanra, 6),
            "DEJ2000": round(meandec, 6),
            "W1mag": round(random.uniform(10, 15), 3),
            "W2mag": round(random.uniform(10, 15), 3),
            "W3mag": round(random.uniform(10, 15), 3),
            "W4mag": round(random.uniform(10, 15), 3),
            "Jmag": round(random.uniform(10, 15), 3),
            "Hmag": round(random.uniform(10, 15), 3),
            "Kmag": round(random.uniform(10, 15), 3),
            "e_W1mag": round(random.uniform(0, 1), 3),
            "e_W2mag": round(random.uniform(0, 1), 3),
            "e_W3mag": round(random.uniform(0, 1), 3),
            "e_W4mag": round(random.uniform(0, 1), 3),
            "e_Jmag": round(random.uniform(0, 1), 3),
            "e_Hmag": round(random.uniform(0, 1), 3),
            "e_Kmag": round(random.uniform(0, 1), 3),
        }
    }
    return fake


def get_fake_empty_xmatch(messages: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=["oid_in", "ra_in", "dec_in", "col1", "aid_in"]
    )
