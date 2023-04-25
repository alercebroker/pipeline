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
            "pid": random.randint(1, 999999),
            "diffmaglim": random.uniform(15, 21),
            "nid": random.randint(1, 999999),
            "magpsf": random.random(),  #
            "sigmapsf": random.random(),  #
            "magap": random.random(),
            "sigmagap": random.random(),
            "magapbig": random.random(),
            "sigmagapbig": random.random(),
        }


def generate_alert_atlas(num_messages: int, identifier: int) -> List[dict]:
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
            "rbversion": f"v7379812",
            "aid": f"AL2X{random.randint(1000, 9990)}",
            "extra_fields": get_extra_fields("ATLAS"),
        }
        alerts.append(alert)
    return alerts


def generate_alert_ztf(num_messages: int, identifier: int) -> List[dict]:
    alerts = []
    for i in range(num_messages):
        alert = {
            "oid": f"ZTFoid{identifier}",
            "aid": f"ZTFaid{identifier}",
            "tid": "ZTF",
            "candid": random.randint(1000000, 9000000),
            "mjd": random.uniform(59000, 60000),
            "fid": random.randint(1, 2),
            "ra": random.uniform(0, 360),
            "dec": random.uniform(-90, 90),
            "e_ra": random.uniform(0, 1),
            "e_dec": random.uniform(0, 1),
            "mag": random.uniform(15, 20),
            "e_mag": random.uniform(0, 1),
            "isdiffpos": random.choice([-1, 1]),
            "rb": random.uniform(0, 1),
            "rbversion": f"v1",
            "extra_fields": get_extra_fields("ZTF"),
            "corrected": random.choice([True, False]),
            "dubious": random.choice([True, False]),
            "has_stamp": random.choice([True, False]),
            "step_id_corr": "test_version",
        }
        alerts.append(alert)
    return alerts


def generate_non_det(num: int, identifier: int) -> List[dict]:
    non_det = []
    for i in range(num):
        nd = {
            "tid": "ZTF",
            "oid": f"ZTFoid{identifier}",
            "aid": f"ZTFaid{identifier}",
            "mjd": random.uniform(59000, 60000),
            "diffmaglim": random.uniform(15, 20),
            "fid": random.randint(1, 2),
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
        candid = int(str(m + 1).ljust(8, "0"))
        detections[-1]["candid"] = candid
        msg = {
            "aid": f"AL2X{str(m).zfill(5)}",
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
            "aid": f"AL2X{str(m).zfill(5)}",
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
    df = pd.DataFrame(messages)
    for i, f in df.iterrows():
        d = {
            "angDist": round(random.uniform(0, 1), 6),
            "col1": random.randint(7, 10),
            "aid_in": f["aid"],
            "ra_in": round(f["meanra"], 6),
            "dec_in": round(f["meandec"], 6),
            "AllWISE": f"J{random.randint(200000, 299999)}.32+240338.4",
            "RAJ2000": round(f["meanra"], 6),
            "DEJ2000": round(f["meandec"], 6),
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
        columns=["oid_in", "ra_in", "dec_in", "col1", "aid_in"]
    )
