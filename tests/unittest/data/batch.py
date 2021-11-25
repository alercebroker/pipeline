import random
import pandas as pd

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
