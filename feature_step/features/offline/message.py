"""Build the magstats_ms_step ZTF output message ("correction-ztf") from DB
reader frames — the exact input contract feature_step consumes.

Schema: features_ztf/schemas/ztf_correction.avsc. Forced photometry is emitted
inside `detections` with forced=True (no separate forced array). Per-epoch aux
fields (rb/procstatus/reference/PS1) go in the extra_fields map. See
docs/superpowers/plans/2026-06-07-features-ztf-stage-3-db-to-message.md.
"""
import numpy as np
import pandas as pd

PS1_KEYS = ["sgscore1", "sgmag1", "srmag1", "simag1", "szmag1", "distpsnr1"]
ZTF_TID = 0  # placeholder telescope id; not read by feature extraction


def _py(v):
    """numpy/NaN/pd.NA -> Avro-friendly python scalar (nulls -> None)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass  # non-scalar (array/list) — leave as-is
    if hasattr(v, "item"):
        v = v.item()
    return v


def _alert(row, forced: bool, extra: dict) -> dict:
    mag_corr = _py(row.get("mag_corr"))
    mag_corr = None if mag_corr is None else float(mag_corr)
    e_mag_corr_ext = _py(row.get("e_mag_corr_ext"))
    return {
        "oid": int(row["oid"]),
        "sid": int(row["sid"]),
        "pid": 0,
        "tid": ZTF_TID,
        "band": int(row["band"]),
        "measurement_id": int(row["measurement_id"]),
        "mjd": float(row["mjd"]),
        "ra": float(row["ra"]),
        "e_ra": None,
        "dec": float(row["dec"]),
        "e_dec": None,
        "mag": float(row["mag"]),
        "e_mag": float(row["e_mag"]),
        "mag_corr": mag_corr,
        "e_mag_corr": None,
        "e_mag_corr_ext": None if e_mag_corr_ext is None else float(e_mag_corr_ext),
        "isdiffpos": int(row["isdiffpos"]),
        "corrected": mag_corr is not None,
        "dubious": False,
        "stellar": False,
        "has_stamp": not forced,
        "forced": forced,
        "new": False,
        "parent_candid": None,
        "extra_fields": {k: _py(v) for k, v in extra.items()},
    }


def build_message(oid, detections: pd.DataFrame, forced: pd.DataFrame, ps1: pd.DataFrame) -> dict:
    """DB reader frames for one oid -> correction-ztf message dict."""
    ps1_extra = {}
    if ps1 is not None and len(ps1):
        ps1_extra = {k: ps1.iloc[0].get(k) for k in PS1_KEYS}

    alerts, meas_ids, ras, decs = [], [], [], []
    if detections is not None and len(detections):
        # to_dict("records") not iterrows(): iterrows unifies row dtype to float64, corrupting >2^53 oid/measurement_id.
        for r in detections.to_dict("records"):
            extra = {"rb": r.get("rb"), "distnr": r.get("distnr"),
                     "rfid": r.get("rfid"), **ps1_extra}
            alerts.append(_alert(r, forced=False, extra=extra))
            meas_ids.append(int(r["measurement_id"]))
            ras.append(float(r["ra"]))
            decs.append(float(r["dec"]))
    if forced is not None and len(forced):
        # to_dict("records") not iterrows(): iterrows unifies row dtype to float64, corrupting >2^53 oid/measurement_id.
        for r in forced.to_dict("records"):
            extra = {"procstatus": r.get("procstatus"), "distnr": r.get("distnr"),
                     "rfid": r.get("rfid"), "sharpnr": r.get("sharpnr"),
                     "chinr": r.get("chinr")}
            alerts.append(_alert(r, forced=True, extra=extra))

    return {
        "oid": int(oid),
        "measurement_id": meas_ids,
        "meanra": float(np.mean(ras)) if ras else 0.0,
        "meandec": float(np.mean(decs)) if decs else 0.0,
        "detections": alerts,
        "non_detections": [],
    }
