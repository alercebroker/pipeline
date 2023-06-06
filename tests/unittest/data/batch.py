import random
import string
import pandas as pd
from typing import List

random.seed(1313)


def generate_batch_ra_dec(n: int, nearest: int = 0) -> pd.DataFrame:
    batch = []
    for i in range(n):
        alert = {
            "oid": f"ALERT{i}",
            "ra": random.uniform(0, 360),
            "dec": random.uniform(-90, 90),
        }
        batch.append(alert)
    for i in range(nearest):
        batch_object = random.randint(0, n - 1)
        alert = batch[batch_object].copy()
        alert["oid"] = f"{alert['oid']}_{i}"
        alert["ra"] = alert["ra"] + random.uniform(-0.000001, 0.000001)
        alert["dec"] = alert["dec"] + random.uniform(-0.000001, 0.000001)
        batch.append(alert)
    batch = pd.DataFrame(batch)
    return batch


def generate_parsed_batch(n: int, nearest: int = 0) -> pd.DataFrame:
    batch = []
    telescopes = ["ATLAS", "ZTF"]
    for i in range(n):
        tid = telescopes[i % 2]
        alert = {
            "oid": f"{tid}_ALERT{i}",
            "tid": tid,
            "candid": str(random.randint(1000000, 9000000)),
            "mjd": random.uniform(57000, 60000),
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
        }
        batch.append(alert)
    for i in range(nearest):
        batch_object = random.randint(0, n - 1)
        alert = batch[batch_object].copy()
        alert["oid"] = f"{alert['oid']}_{i}"
        alert["ra"] = alert["ra"] + random.uniform(-0.000001, 0.000001)
        alert["dec"] = alert["dec"] + random.uniform(-0.000001, 0.000001)
        batch.append(alert)
    batch = pd.DataFrame(batch)
    return batch


def _generate_ztf_batch(n: int, nearest: int = 0) -> List[dict]:
    batch = []
    for i in range(n):
        m = random.randrange(0, 30)  # m is the quantity of prv_candidates
        alert = {
            "objectId": f"ZTF_ALERT{i}",
            "publisher": "ZTF",
            "cutoutScience": {"stampData": b"science"},
            "cutoutTemplate": {"stampData": b"template"},
            "cutoutDifference": {"stampData": b"difference"},
            "candidate": {
                "jd": random.randrange(2458000, 2459000),
                "ra": random.uniform(0, 360),
                "dec": random.uniform(-90, 90),
                "magpsf": random.uniform(15, 20),
                "sigmapsf": random.random(),
                "fid": random.randint(1, 2),
                "candid": str(random.randint(1000000, 9000000)),
                "pid": random.randint(1000000, 9000000),
                "rfid": random.randint(1000000, 9000000),
                "isdiffpos": random.choice(["t", "f", "1", "0"]),
                "rb": random.random(),
                "rbversion": f"v1",
            },
            "prv_candidates": random.choice(
                [
                    None,
                    [
                        {
                            "candid": random.choice(
                                [None, str(random.randint(1000000, 9000000))]
                            ),
                            "jd": random.randrange(2458000, 2459000),
                            "fid": random.randint(1, 2),
                            "rfid": random.randint(1000000, 9000000),
                            "ra": random.uniform(0, 360),
                            "dec": random.uniform(-90, 90),
                            "magpsf": random.uniform(15, 20),
                            "sigmapsf": random.random(),
                            "isdiffpos": random.choice(["t", "f", "1", "0"]),
                            "rb": random.random(),
                            "rbversion": f"v1",
                        }
                        for _ in range(0, m)
                    ],
                ]
            ),
        }
        batch.append(alert)
    for i in range(nearest):
        batch_object = random.randint(0, n - 1)
        alert = batch[batch_object].copy()
        alert["objectId"] = f"{alert['objectId']}_{i}"
        alert["candidate"]["ra"] = alert["candidate"]["ra"] + random.uniform(
            -0.000001, 0.000001
        )
        alert["candidate"]["dec"] = alert["candidate"]["dec"] + random.uniform(
            -0.000001, 0.000001
        )
        batch.append(alert)
    return batch


def _generate_atlas_batch(n: int, nearest: int = 0) -> List[dict]:
    batch = []
    for i in range(n):
        alert = {
            "objectId": f"ATLAS_ALERT{i}",
            "publisher": "ATLAS",
            "cutoutScience": {"stampData": b"science"},
            "cutoutDifference": {"stampData": b"difference"},
            "candidate": {
                "jd": random.randrange(2458000, 2459000),
                "ra": random.uniform(0, 360),
                "dec": random.uniform(-90, 90),
                "magpsf": random.uniform(15, 20),
                "sigmapsf": random.random(),
                "fid": random.randint(1, 2),
                "candid": "".join(
                    random.choices(string.ascii_letters + string.digits, k=30)
                ),
                "pid": random.randint(1000000, 9000000),
                "rfid": random.randint(1000000, 9000000),
                "isdiffpos": random.choice(["t", "f"]),
                "rb": random.random(),
                "rbversion": f"v1",
                "mjd": random.uniform(57000, 60000),
                "RA": random.uniform(0, 360),
                "Dec": random.uniform(-90, 90),
                "Mag": random.uniform(15, 20),
                "Dmag": random.random(),
                "filter": "o",
            },
        }
        batch.append(alert)
    for i in range(nearest):
        batch_object = random.randint(0, n - 1)
        alert = batch[batch_object].copy()
        alert["objectId"] = f"{alert['objectId']}_{i}"
        alert["candidate"]["RA"] = alert["candidate"]["RA"] + random.uniform(
            -0.000001, 0.000001
        )
        alert["candidate"]["RA"] = alert["candidate"]["RA"] + random.uniform(
            -0.000001, 0.000001
        )
        batch.append(alert)
    return batch


def _generate_elasticc_batch(n: int) -> List[dict]:
    def gen_elasticc_object():
        return {
            "diaSourceId": random.randint(1000000, 9000000),
            "diaObjectId": random.choice([None, random.randint(1000000, 9000000)]),
            "midPointTai": random.uniform(57000, 60000),
            "filterName": "fid",
            "ra": random.uniform(0, 360),
            "decl": random.uniform(-90, 90),
        }

    batch = []



def random_sub_samples(samples: int, size: int):
    sequence = []
    if samples == 0:
        sequence = [0 for _ in range(size)]
    else:
        for _ in range(size):
            if _ == size - 1:
                val = samples - sum(sequence)
            else:
                remain = samples - sum(sequence)
                val = random.randrange(0, remain)
            sequence.append(val)
    return sequence


def generate_alerts_batch(n: int, nearest: int = 0) -> List[dict]:
    generators = [_generate_ztf_batch, _generate_atlas_batch]
    sub_samples = random_sub_samples(n, len(generators))
    sub_nearest = random_sub_samples(nearest, len(generators))
    batch = []
    for generator, m, near in zip(generators, sub_samples, sub_nearest):
        b = generator(m, nearest=near)
        batch.append(b)
    batch = sum(batch, [])
    return batch
