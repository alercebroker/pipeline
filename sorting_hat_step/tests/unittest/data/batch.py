import random
import string
import pandas as pd
from typing import List
from datetime import datetime

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
            "rbversion": "v1",
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
                "ndethist": random.randint(5, 15),
                "ncovhist": random.randint(5, 15),
                "jdstarthist": random.randint(2458000, 2459000),
                "jdendhist": random.randint(2458000, 2459000),
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
                "rbversion": "v1",
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
                            "rbversion": "v1",
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
                "rbversion": "v1",
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


def _generate_elasticc_batch(n: int, nearest: int = 0) -> List[dict]:
    def gen_elasticc_object():
        dia_source = {
            "diaSourceId": random.randint(1000000, 9000000),
            "diaObjectId": random.randint(1000000, 9000000),
            "midPointTai": random.uniform(57000, 60000),
            "filterName": "fid",
            "ra": random.uniform(0, 360),
            "decl": random.uniform(-90, 90),
            "psFlux": random.uniform(15, 30),
            "psFluxErr": random.random(),
            "snr": random.random(),
        }
        return {
            "alertId": random.randint(1000000, 9000000),
            "diaSource": dia_source,
            "prvDiaSources": [dia_source],
            "prvDiaForcedSources": [],
            "diaObject": {
                "diaObjectId": 498,
                "ra": 296.95755,
                "decl": -50.480045,
                "mwebv": 0.0,
                "mwebv_err": 0.0,
                "z_final": 0.0,
                "z_final_err": 0.0,
                "hostgal_ellipticity": 0.0,
                "hostgal_sqradius": 0.0,
                "hostgal_z": 1.3332124948501587,
                "hostgal_z_err": 0.0010000000474974513,
                "hostgal_zphot_q10": 0.40760189294815063,
                "hostgal_zphot_q20": 0.26571303606033325,
                "hostgal_zphot_q30": 0.31939709186553955,
                "hostgal_zphot_q40": 0.06509595364332199,
                "hostgal_zphot_q50": 0.4050394892692566,
                "hostgal_zphot_q60": 0.47056853771209717,
                "hostgal_zphot_q70": 0.48318788409233093,
                "hostgal_zphot_q80": 0.734911322593689,
                "hostgal_zphot_q90": 0.8748581409454346,
                "hostgal_zphot_q99": 0.6080126762390137,
                "hostgal_mag_u": 28.53758430480957,
                "hostgal_mag_g": 28.049013137817383,
                "hostgal_mag_r": 27.611717224121094,
                "hostgal_mag_i": 26.858409881591797,
                "hostgal_mag_z": 26.330713272094727,
                "hostgal_mag_Y": 25.54668617248535,
                "hostgal_ra": 0.0,
                "hostgal_dec": 0.0,
                "hostgal_snsep": 0.019439058378338814,
                "hostgal_magerr_u": -999.0,
                "hostgal_magerr_g": -999.0,
                "hostgal_magerr_r": -999.0,
                "hostgal_magerr_i": -999.0,
                "hostgal_magerr_z": -999.0,
                "hostgal_magerr_Y": -999.0,
                "hostgal2_ellipticity": 0.0,
                "hostgal2_sqradius": 0.0,
                "hostgal2_z": -9.0,
                "hostgal2_z_err": -9.0,
                "hostgal2_zphot_q10": -9.0,
                "hostgal2_zphot_q20": -9.0,
                "hostgal2_zphot_q30": -9.0,
                "hostgal2_zphot_q40": -9.0,
                "hostgal2_zphot_q50": -9.0,
                "hostgal2_zphot_q60": -9.0,
                "hostgal2_zphot_q70": -9.0,
                "hostgal2_zphot_q80": -9.0,
                "hostgal2_zphot_q90": -9.0,
                "hostgal2_zphot_q99": -9.0,
                "hostgal2_mag_u": 999.0,
                "hostgal2_mag_g": 999.0,
                "hostgal2_mag_r": 999.0,
                "hostgal2_mag_i": 999.0,
                "hostgal2_mag_z": 999.0,
                "hostgal2_mag_Y": 999.0,
                "hostgal2_ra": 0.0,
                "hostgal2_dec": 0.0,
                "hostgal2_snsep": -9.0,
                "hostgal2_magerr_u": 999.0,
                "hostgal2_magerr_g": 999.0,
                "hostgal2_magerr_r": 999.0,
                "hostgal2_magerr_i": 999.0,
                "hostgal2_magerr_z": 999.0,
                "hostgal2_magerr_Y": 999.0,
            },
        }

    batch = [gen_elasticc_object() for _ in range(n)]
    for _ in range(nearest):
        near = random.choice(batch)
        near = near.copy()

        near["alertId"] = str(near["alertId"]) + "near"
        near["diaSource"]["ra"] = near["diaSource"]["ra"] + random.uniform(
            -0.000001, 0.000001
        )
        near["diaSource"]["decl"] = near["diaSource"]["decl"] + random.uniform(
            -0.000001, 0.000001
        )
        batch.append(near)

    return batch


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
    generators = [_generate_ztf_batch, _generate_atlas_batch, _generate_elasticc_batch]
    sub_samples = random_sub_samples(n, len(generators))
    sub_nearest = random_sub_samples(nearest, len(generators))
    batch = []
    for generator, m, near in zip(generators, sub_samples, sub_nearest):
        b = generator(m, nearest=near)
        b = list(
            map(
                lambda el: {
                    **el,
                    "timestamp": int(datetime.now().timestamp()),
                    "topic": "topik",
                },
                b,
            )
        )
        batch.append(b)
    batch = sum(batch, [])
    return batch
