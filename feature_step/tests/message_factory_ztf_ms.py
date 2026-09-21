"""Build `magstats_ms_ztf` messages (schemas/magstats_ms_step/ztf/output.avsc).

Flat records, no `extra_fields`. Three arrays with distinct vocabularies:
`detections` (candidate), `previous_detections` (prv_candidate) and
`forced_photometries` (forced_photometry).
"""
import random

ZTF_SID = 0
ZTF_TID = 0
BAND_MAP = {"g": 1, "r": 2, "i": 3}


def _base_epoch(oid, measurement_id, band, mjd, forced, rng):
    return {
        "new": True,
        "oid": oid,
        "sid": ZTF_SID,
        "tid": ZTF_TID,
        "pid": rng.randint(1, 999999),
        "band": BAND_MAP[band],
        "measurement_id": measurement_id,
        "mjd": mjd,
        "ra": 250.0 + rng.random() * 1e-4,
        "e_ra": 0.0001,
        "dec": 30.0 + rng.random() * 1e-4,
        "e_dec": 0.0001,
        "mag": 18.0 + rng.random(),
        "e_mag": 0.05 + rng.random() * 0.05,
        "isdiffpos": 1,
        "forced": forced,
        "parent_candid": None,
        "corrected": True,
        "dubious": False,
        "stellar": False,
        "diffmaglim": 20.5,
    }


def candidate(oid, measurement_id, band, mjd, rng, rb=0.9, rfid=783120150):
    """One `candidate` record: rb, PS1 columns, magpsf_corr/sigmapsf_corr_ext."""
    epoch = _base_epoch(oid, measurement_id, band, mjd, False, rng)
    epoch.update(
        {
            "has_stamp": True,
            "magpsf_corr": 17.5 + rng.random(),
            "sigmapsf_corr": 0.05,
            "sigmapsf_corr_ext": 0.06 + rng.random() * 0.02,
            "rb": rb,
            "distnr": 0.3 + rng.random() * 0.2,
            "magnr": 17.0,
            "sigmagnr": 0.02,
            "chinr": 0.5,
            "sharpnr": -0.02,
            "rfid": rfid,
            "sgscore1": 0.1,
            "sgmag1": 18.1,
            "srmag1": 17.9,
            "simag1": 17.8,
            "szmag1": 17.7,
            "distpsnr1": 0.4,
            "rbversion": "t17_f5_c3",
            "drbversion": "d6_m7",
        }
    )
    return epoch


def prv_candidate(oid, measurement_id, band, mjd, rng, rb=0.9):
    """One `prv_candidate` record: rb but no rfid and no PS1 columns."""
    epoch = _base_epoch(oid, measurement_id, band, mjd, False, rng)
    epoch.update(
        {
            "has_stamp": False,
            "magpsf_corr": 17.5 + rng.random(),
            "sigmapsf_corr": 0.05,
            "sigmapsf_corr_ext": 0.06 + rng.random() * 0.02,
            "rb": rb,
            "distnr": 0.3 + rng.random() * 0.2,
            "magnr": 17.0,
            "sigmagnr": 0.02,
            "chinr": 0.5,
            "sharpnr": -0.02,
            "rbversion": "t17_f5_c3",
        }
    )
    return epoch


def forced_photometry(oid, measurement_id, band, mjd, rng, procstatus="0",
                      rfid=783120150):
    """One `forced_photometry` record: procstatus, mag_corr/e_mag_corr_ext, no rb."""
    epoch = _base_epoch(oid, measurement_id, band, mjd, True, rng)
    epoch.update(
        {
            "mag_corr": 17.5 + rng.random(),
            "e_mag_corr": 0.05,
            "e_mag_corr_ext": 0.06 + rng.random() * 0.02,
            "procstatus": procstatus,
            "distnr": 0.3 + rng.random() * 0.2,
            "magnr": 17.0,
            "sigmagnr": 0.02,
            "chinr": 0.5,
            "sharpnr": -0.02,
            "rfid": rfid,
            "ranr": 250.0,
            "decnr": 30.0,
            "programid": 1,
            "forcediffimflux": 100.0,
            "forcediffimfluxunc": 10.0,
        }
    )
    return epoch


def non_detection(oid, band, mjd):
    return {
        "oid": oid,
        "sid": ZTF_SID,
        "tid": ZTF_TID,
        "band": BAND_MAP[band],
        "mjd": mjd,
        "diffmaglim": 20.5,
    }


def generate_message(
    oid=36028941624528297,
    bands=("g", "r"),
    n_detections=6,
    n_previous_detections=4,
    n_forced=5,
    seed=42,
    with_xmatch=False,
):
    rng = random.Random(seed)
    mjd = 60000.0
    mid = 1000

    detections, previous_detections, forced = [], [], []
    for i in range(n_detections):
        band = bands[i % len(bands)]
        detections.append(candidate(oid, mid, band, mjd, rng))
        mjd += 1.7
        mid += 1
    for i in range(n_previous_detections):
        band = bands[i % len(bands)]
        previous_detections.append(prv_candidate(oid, mid, band, mjd, rng))
        mjd += 1.7
        mid += 1
    for i in range(n_forced):
        band = bands[i % len(bands)]
        forced.append(forced_photometry(oid, mid, band, mjd, rng))
        mjd += 1.7
        mid += 1

    message = {
        "oid": oid,
        "sid": ZTF_SID,
        "measurement_id": [d["measurement_id"] for d in detections],
        "meanra": 250.0,
        "meandec": 30.0,
        "detections": detections,
        "previous_detections": previous_detections,
        "forced_photometries": forced,
        "non_detections": [non_detection(oid, "g", 59990.0)],
    }
    if with_xmatch:
        message["xmatches"] = allwise_match(oid)
    return message


def allwise_match(oid, w1=15.1, w2=14.9, w3=12.5, w4=9.1):
    """The shape `XmatchClient.conesearch_with_metadata` returns."""
    return {
        "oid": str(oid),
        "catalog": "allwise",
        "distance": 0.5,
        "match_id": "J000000.00+000000.0",
        "metadata": {
            "w1mpro": {"Float64": w1},
            "w2mpro": {"Float64": w2},
            "w3mpro": {"Float64": w3},
            "w4mpro": {"Float64": w4},
        },
    }
