import random

from typing import List

random.seed(8798, version=2)


def get_extra_fields(telescope: str):
    if telescope == "ATLAS":
        return {}
    elif telescope == "ZTF":
        return {
            "distnr": random.random(),
            "magnr": random.random(),
            "sigmagnr": random.random(),
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
            "tid": "ZTF",
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
            "rbversion": f"v1",
            "extra_fields": get_extra_fields("ZTF"),
        }
        alerts.append(alert)
    return alerts


def generate_non_det(num: int, identifier: int) -> List[dict]:
    non_det = []
    for i in range(num):
        nd = {
            "tid": "ZTF",
            "oid": f"ZTFoid{identifier}",
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
        non_det = generate_non_det(random.randint(0, 20), m)
        msg = {
            "aid": f"AL2X{str(m).zfill(5)}",
            "candid": int(str(m + 1).ljust(8, "0")),
            "meanra": random.uniform(0, 360),
            "meandec": random.uniform(-90, 90),
            "detections": detections,
            "non_detections": non_det,
            "ndet": len(detections),
        }
        batch.append(msg)
    random.shuffle(batch, lambda: 0.1)
    return batch
