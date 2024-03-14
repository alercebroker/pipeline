import random

from typing import List

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
            "diffmaglim": random.uniform(15, 21),
            "nid": random.randint(1, 999999),
            "magpsf": random.random(),  #
            "sigmapsf": random.random(),  #
            "magap": random.random(),
            "sigmagap": random.random(),
            "magapbig": random.random(),
            "sigmagapbig": random.random(),
        }


def generate_alert(sid, tid, identifier):
    alert = {
        "oid": f"{sid}oid{identifier}",
        "tid": tid,
        "sid": sid,
        "pid": random.randint(1, 999999),
        "candid": str(random.randint(1000000, 9000000)),
        "mjd": random.uniform(59000, 60000),
        "fid": random.choice(["o", "c"]),
        "ra": random.uniform(0, 360),
        "dec": random.uniform(-90, 90),
        "e_ra": random.random(),
        "e_dec": random.random(),
        "mag": random.uniform(15, 20),
        "e_mag": random.random(),
        "mag_corr": random.uniform(15, 20),
        "e_mag_corr": random.random(),
        "e_mag_corr_ext": random.random(),
        "isdiffpos": random.choice([-1, 1]),
        "aid": f"AL2X{random.randint(1000, 9990)}",
        "corrected": random.choice([True, False]),
        "dubious": random.choice([True, False]),
        "stellar": random.choice([True, False]),
        "has_stamp": random.choice([True, False]),
        "forced": random.choice([True, False]),
        "new": random.choice([True, False]),
        "extra_fields": get_extra_fields(sid),
        "parent_candid": random.choice(
            [None, random.randint(1000000, 9000000)]
        ),
    }
    return alert


def generate_alert_atlas(num_messages: int, identifier: int) -> List[dict]:
    alerts = []
    for i in range(num_messages):
        alert = generate_alert("ATLAS", "ATLAS-01a", identifier)
        alerts.append(alert)
    return alerts


def generate_alert_ztf(num_messages: int, identifier: int) -> List[dict]:
    alerts = []
    for i in range(num_messages):
        alert = generate_alert("ZTF", "ZTF", identifier)
        alerts.append(alert)
    return alerts


def generate_non_det(num: int, identifier: int) -> List[dict]:
    non_det = []
    for i in range(num):
        nd = {
            "tid": "ZTF",
            "sid": "ZIF",
            "oid": f"ZTFoid{identifier}",
            "aid": f"ZTFaid{identifier}",
            "mjd": random.uniform(59000, 60000),
            "diffmaglim": random.uniform(15, 20),
            "fid": random.choice(["g", "r"]),
        }
        non_det.append(nd)
    return non_det


def generate_input_batch(n: int) -> List[dict]:
    """
    Parameters
    ----------
    n: number of objects in the batch
    Returns a batch of generic message
    -------
    """
    batch = []
    for m in range(1, n + 1):
        detections = generate_alert_atlas(
            random.randint(1, 100), m
        ) + generate_alert_ztf(random.randint(1, 100), m)
        non_det = generate_non_det(random.randint(1, 20), m)
        candid = str(m + 1).ljust(8, "0")
        detections[-1]["candid"] = candid
        msg = {
            "oid": f"XX{str(m).zfill(5)}",
            "candid": [candid],
            "meanra": random.uniform(0, 360),
            "meandec": random.uniform(-90, 90),
            "detections": detections,
            "non_detections": non_det,
        }
        batch.append(msg)
    random.shuffle(batch, lambda: 0.1)
    return batch


def generate_non_ztf_batch(n: int) -> List[dict]:
    batch = []
    for m in range(1, n + 1):
        detections = generate_alert_atlas(random.randint(1, 100), m)
        non_det = generate_non_det(random.randint(1, 20), m)
        candid = int(str(m + 1).ljust(8, "0"))
        detections[-1]["candid"] = candid
        msg = {
            "oid": f"XX{str(m).zfill(5)}",
            "candid": [candid],
            "sid": ["ATLAS"],
            "meanra": random.uniform(0, 360),
            "meandec": random.uniform(-90, 90),
            "detections": detections,
            "non_detections": non_det,
        }
        batch.append(msg)
    random.shuffle(batch, lambda: 0.1)
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


def get_fake_xmatch(messages: List[dict]) -> pd.DataFrame:
    fake = []
    for m in messages:
        d = {
            "angDist": round(random.uniform(0, 1), 6),
            "col1": random.randint(7, 10),
            "oid_in": m["oid"],
            "ra_in": round(m["meanra"], 6),
            "dec_in": round(m["meandec"], 6),
            "AllWISE": f"J{random.randint(200000, 299999)}.32+240338.4",
            "RAJ2000": round(m["meanra"], 6),
            "DEJ2000": round(m["meandec"], 6),
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
        fake.append(d)
    return pd.DataFrame(fake)


def get_fake_empty_xmatch(messages: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=["oid_in", "ra_in", "dec_in", "col1", "oid_in"]
    )
