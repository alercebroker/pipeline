import random
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
        batch_object = random.randint(0, n-1)
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
            "candid": random.randint(1000000, 9000000),
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
        batch_object = random.randint(0, n-1)
        alert = batch[batch_object].copy()
        alert["oid"] = f"{alert['oid']}_{i}"
        alert["ra"] = alert["ra"] + random.uniform(-0.000001, 0.000001)
        alert["dec"] = alert["dec"] + random.uniform(-0.000001, 0.000001)
        batch.append(alert)
    batch = pd.DataFrame(batch)
    return batch


def generate_alerts_batch(n: int, nearest: int = 0) -> List[dict]:
    batch = []
    telescopes = ["ATLAS", "ZTF"]
    for i in range(n):
        tid = telescopes[i % 2]
        alert = {
            "objectId": f"{tid}_ALERT{i}",
            "publisher": tid,
            "cutoutScience": {"stampData": b""},
            "cutoutTemplate": {"stampData": b""},
            "cutoutDifference": {"stampData": b""},
            "candidate": {
                "candid": random.randint(1000000, 9000000),
                "pid": random.randint(1000000, 9000000),
                "rfid": random.randint(1000000, 9000000),
                "isdiffpos": random.choice(["t", "f", "1", "0"]),
                "rb": random.random(),
                "rbversion": f"v1",
            },
        }
        if tid == "ATLAS":
            alert["candidate"]["mjd"] = random.uniform(57000, 60000)
            alert["candidate"]["RA"] = random.uniform(0, 360)
            alert["candidate"]["Dec"] = random.uniform(-90, 90)
            alert["candidate"]["Mag"] = random.uniform(15, 20)
            alert["candidate"]["Dmag"] = random.random()
            alert["candidate"]["filter"] = "o"
            pass
        elif tid == "ZTF":
            alert["candidate"]["jd"] = random.randrange(2458000, 2459000)
            alert["candidate"]["ra"] = random.uniform(0, 360)
            alert["candidate"]["dec"] = random.uniform(-90, 90)
            alert["candidate"]["magpsf"] = random.uniform(15, 20)
            alert["candidate"]["sigmapsf"] = random.random()
            alert["candidate"]["fid"] = random.randint(1, 2)
            alert["prv_candidates"] = {
                "candid": random.choice([None, random.randint(1000000, 9000000)]),
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
        batch.append(alert)
    for i in range(nearest):
        batch_object = random.randint(0, n-1)
        alert = batch[batch_object].copy()
        alert["oid"] = f"{alert['oid']}_{i}"
        alert["ra"] = alert["ra"] + random.uniform(-0.000001, 0.000001)
        alert["dec"] = alert["dec"] + random.uniform(-0.000001, 0.000001)
        batch.append(alert)
    return batch
