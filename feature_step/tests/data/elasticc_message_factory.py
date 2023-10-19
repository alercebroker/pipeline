import random
from test_utils.mockdata.extra_fields.elasticc import generate_extra_fields

random.seed(8798, version=2)

ELASTICC_BANDS = ["u", "g", "r", "i", "z", "Y"]


def get_extra_fields():
    extra_fields = generate_extra_fields()
    extra_fields.update(
        {
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
    )
    return extra_fields


def generate_alert(
    aid: str, band: str, num_messages: int, identifier: int, **kwargs
) -> list[dict]:
    alerts = []
    survey_id = kwargs.get("survey", "LSST")
    diaObject = get_extra_fields()["diaObject"]
    for i in range(num_messages):
        extra_fields = get_extra_fields()
        extra_fields["diaObject"] = diaObject
        alert = {
            "candid": str(random.randint(1000000, 9000000)),
            "oid": f"oid{identifier}",
            "aid": aid,
            "tid": survey_id,
            "mjd": random.uniform(59000, 60000),
            "sid": survey_id,
            "fid": band,
            "pid": random.randint(1000000, 9000000),
            "ra": random.uniform(-90, 90),
            "e_ra": random.uniform(-90, 90),
            "dec": random.uniform(-90, 90),
            "e_dec": random.uniform(-90, 90),
            "mag": random.uniform(15, 20),
            "e_mag": random.uniform(0, 1),
            "mag_corr": kwargs.get("mag_corr")
            if "mag_corr" in kwargs.keys()
            else random.uniform(15, 20),
            "e_mag_corr": random.uniform(0, 1),
            "e_mag_corr_ext": kwargs.get("e_mag_corr_ext")
            if "e_mag_corr_ext" in kwargs
            else random.uniform(0, 1),
            "isdiffpos": random.choice([-1, 1]),
            "corrected": kwargs.get("corrected")
            if "corrected" in kwargs
            else random.choice([True, False]),
            "dubious": random.choice([True, False]),
            "has_stamp": random.choice([True, False]),
            "stellar": random.choice([True, False]),
            "new": random.choice([True, False]),
            "forced": random.choice([True, False]),
            "parent_candid": random.randint(1000000, 9000000),
            "extra_fields": extra_fields,
        }
        alerts.append(alert)
    return alerts


def generate_non_det(aid: str, num: int, identifier: int) -> list[dict]:
    non_det = []
    for i in range(num):
        nd = {
            "aid": aid,
            "tid": "LSST",
            "oid": f"LSSToid{identifier}",
            "sid": "LSST",
            "mjd": random.uniform(59000, 60000),
            "fid": random.choice(ELASTICC_BANDS),
            "diffmaglim": random.uniform(15, 20),
        }
        non_det.append(nd)
    return non_det


def generate_input_batch(
    n: int, bands: list[str], offset=0, survey="LSST"
) -> list[dict]:
    """
    Parameters
    ----------
    n: number of objects in the batch
    Returns a batch of generic message
    -------
    """
    batch = []
    for m in range(1, n + 1):
        aid = f"AL2X{str(m+offset).zfill(5)}"
        meanra = random.uniform(0, 360)
        meandec = random.uniform(-90, 90)
        detections = []
        for band in bands:
            detections.extend(
                generate_alert(
                    aid, band, random.randint(6, 10), m, survey=survey
                )
            )
        non_det = generate_non_det(aid, random.randint(0, 1), m)
        xmatch = {}
        msg = {
            "aid": aid,
            "meanra": meanra,
            "meandec": meandec,
            "detections": detections,
            "non_detections": non_det,
            "xmatches": xmatch,
        }
        batch.append(msg)
    random.sample(batch, len(batch))
    return batch


def generate_bad_emag_ratio(n: int, bands: list[str], offset=0) -> list[dict]:
    """
    Parameters
    ----------
    n: number of objects in the batch
    Returns a batch of generic message
    mag_corr
    e_mag_corr_ext
    -------
    """
    batch = []
    for m in range(1, n + 1):
        aid = f"AL2X{str(m+offset).zfill(5)}"
        meanra = random.uniform(0, 360)
        meandec = random.uniform(-90, 90)
        detections = []
        for band in bands:
            detections.extend(
                generate_alert(
                    aid,
                    band,
                    random.randint(5, 10),
                    m,
                    mag_corr=random.uniform(15, 20),
                    e_mag_corr_ext=random.uniform(10, 15),
                    corrected=True,
                )
            )
        non_det = generate_non_det(aid, random.randint(0, 1), m)
        # candid = int(str(m + 1).ljust(8, "0"))
        # detections[-1]["candid"] = candid
        xmatch = {}
        msg = {
            "aid": aid,
            "meanra": meanra,
            "meandec": meandec,
            "detections": detections,
            "non_detections": non_det,
            "xmatches": xmatch,
        }
        batch.append(msg)
    random.sample(batch, len(batch))
    return batch
