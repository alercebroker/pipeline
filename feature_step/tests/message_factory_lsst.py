import pandas as pd
import random
from test_utils.mockdata.extra_fields.elasticc import generate_extra_fields
from features.utils.parsers import fid_mapper_for_db

random.seed(8798, version=2)

ELASTICC_BANDS = ["u", "g", "r", "i", "z", "y"]
#measurement_id es candid



def generate_alert(oid: str, band: str, num_messages: int, identifier: int, **kwargs) -> list[dict]:
    """
    Generate a list of detections for a given object ID and band.
    """
    alerts = []
    survey_id = kwargs.get("survey", "LSST")

    for i in range(num_messages):
        alert = { #esto tengo que cambiarlo por los campos reales.
            "oid": oid,
            "sid": survey_id,
            "mjd": random.uniform(59000, 60000),
            "ra": random.uniform(0, 360),
            "dec": random.uniform(-90, 90),
            "psfFlux": random.uniform(100, 200),
            "psfFluxErr": random.uniform(1, 5),
            "scienceFlux": random.uniform(200, 300),
            "scienceFluxErr": random.uniform(1, 5),
            "tid": survey_id,
            "band": band,
            "diaObjectId": random.randint(1000000, 9000000),
            'measurement_id': random.randint(1000000, 9000000),
        }
        alerts.append(alert)
    return alerts



def generate_input_batch_lsst(n: int, bands: list[str], offset=0, survey="LSST") -> list[dict]:
    """
    Generate a batch of fake LSST messages with minimal fields.
    """
    batch = []
    for m in range(1, n + 1):
        oid = f"AL2X{str(m+offset).zfill(5)}"
        detections = []
        for band in bands:
            detections.extend(
                generate_alert(oid, band, random.randint(60, 100), m, survey=survey)
            )

        msg = {
            "oid": oid,
            "measurement_id": random.randint(1000000, 9000000),
            "detections": detections,
            "non_detections": [],  
            "xmatches": {} , #esto no deberia ir
        }
        batch.append(msg)
    return batch

