#!/usr/bin/env python
"""Experiment: does using the LEGACY alerce.forced_photometry (instead of the
multisurvey_ztf reprocessed one) reproduce the stored BHRF 2.1.0 prediction?

Keeps detections/ps1/allwise/references from multisurvey_ztf (detections are
byte-identical to legacy), swaps ONLY the forced photometry, and classifies both
ways, comparing each against alerce.probability. Confirms the procstatus 61->0
forced-photometry difference is the cause.

    MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl \
        conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_forced_swap_experiment.py
"""
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]
for p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper",
          PIPE / "libs" / "apf", PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(p))

import pandas as pd
from sqlalchemy import text

from features.offline import db
from features.offline.classify import load_squidward_model, classify_astro_object
from features.offline.lc_features import compute_astro_object
from features.offline.message import build_message
from features.offline.probability_compare import (
    BHRF_CLASSIFIER_NAMES, compare_probability_frames, offline_dto_to_series)

CRED = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")
ZTF, BIG = "ZTF18abkhtlr", 36028941608134451


def fetch_legacy_forced(ztf_oid: str, bigint: int) -> pd.DataFrame:
    """alerce.forced_photometry for one oid, shaped like db.fetch_forced_photometry.

    Carries procstatus/mag AS-IS (incl. the procstatus=61 / mag=100 flagged epochs)
    so the SAME pipeline filters drop them exactly as the legacy prediction did.
    """
    eng = db._make_engine(CRED)
    q = text("""
        SELECT mjd, fid AS band, ra, dec, mag, e_mag, mag_corr,
               e_mag_corr_ext, isdiffpos, procstatus, distnr, rfid, sharpnr, chinr, pid
        FROM alerce.forced_photometry WHERE oid = :o
    """)
    with eng.connect() as c:
        df = pd.read_sql_query(q, c, params={"o": ztf_oid})
    df["oid"] = bigint
    df["sid"] = 0
    df["measurement_id"] = df["pid"].astype("int64")
    df["isdiffpos"] = db.normalize_isdiffpos(df["isdiffpos"])
    return df


def fetch_legacy_detections(ztf_oid: str, bigint: int) -> pd.DataFrame:
    """alerce.detection for one oid, shaped like db.fetch_detections."""
    eng = db._make_engine(CRED)
    q = text("""
        SELECT mjd, fid AS band, ra, dec, magpsf AS mag, sigmapsf AS e_mag,
               magpsf_corr AS mag_corr, sigmapsf_corr_ext AS e_mag_corr_ext,
               isdiffpos, distnr, rb, rfid, candid
        FROM alerce.detection WHERE oid = :o
    """)
    with eng.connect() as c:
        df = pd.read_sql_query(q, c, params={"o": ztf_oid})
    df["oid"] = bigint
    df["sid"] = 0
    df["measurement_id"] = df["candid"].astype("int64")
    df["isdiffpos"] = db.normalize_isdiffpos(df["isdiffpos"])
    return df


def classify_with_forced(model, dets, forced, ps1, refs, allwise):
    message = build_message(BIG, dets, forced, ps1)
    ao = compute_astro_object(message, refs, allwise, min_detections=1)
    if ao is None:
        return None
    return classify_astro_object(ao, message, model)


def report(tag, result, stored):
    if result is None or result.probabilities is None or len(result.probabilities) == 0:
        print(f"\n[{tag}] no probabilities"); return
    off = offline_dto_to_series(result)
    _, s = compare_probability_frames(off, stored, rtol=1e-3, atol=1e-4)
    print(f"\n[{tag}]  passed={s['passed']}  match={s['match']} differ={s['differ']} "
          f"only_off={s['only_offline']} only_stored={s['only_stored']}  "
          f"rank1_agree={s['rank1_agree']}/{s['rank1_total']}")
    for cname, r in s["rank1"].items():
        short = cname.replace("lc_classifier_BHRF_forced_phot_", "").replace(
            "lc_classifier_BHRF_forced_phot", "flat")
        flag = "OK" if r["agree"] else "XX"
        print(f"    {flag} {short:11s} offline={r['offline']!s:10} stored={r['stored']!s:10}")


def main():
    model, name, version = load_squidward_model()
    print(f"model: {name} v{version}")

    dets = db.fetch_detections(CRED, [BIG])
    ps1 = db.fetch_ps1(CRED, [BIG])
    allwise = db.fetch_allwise(CRED, [BIG])
    refs = db.fetch_references(CRED, [BIG])
    ms_forced = db.fetch_forced_photometry(CRED, [BIG])
    legacy_forced = fetch_legacy_forced(ZTF, BIG)
    print(f"forced epochs — multisurvey_ztf: {len(ms_forced)}   legacy alerce: {len(legacy_forced)}")

    stored = db.fetch_stored_probabilities(CRED, ZTF, BHRF_CLASSIFIER_NAMES, "2.1.0")
    print(f"stored BHRF 2.1.0 rows: {len(stored)}")

    legacy_dets = fetch_legacy_detections(ZTF, BIG)
    print(f"detections   — multisurvey_ztf: {len(dets)}   legacy alerce: {len(legacy_dets)}")

    report("CONTROL: ms dets + ms forced",
           classify_with_forced(model, dets, ms_forced, ps1, refs, allwise), stored)
    report("SWAP forced: ms dets + legacy forced",
           classify_with_forced(model, dets, legacy_forced, ps1, refs, allwise), stored)
    report("FULL legacy: legacy dets + legacy forced",
           classify_with_forced(model, legacy_dets, legacy_forced, ps1, refs, allwise), stored)


if __name__ == "__main__":
    main()
