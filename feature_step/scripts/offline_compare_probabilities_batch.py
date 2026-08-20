#!/usr/bin/env python
"""Batch version of offline_compare_probabilities.py: N oids, aggregate agreement.

Runs the offline pipeline (DB -> features -> BHRF) over a stratified sample of
oids produced by offline_sample_classified_oids.py, reads the stored legacy
probabilities for the same oids from `alerce.probability`, and reports aggregate
agreement: per-head rank-1 match rate and the per-class probability difference
distribution.

The stored probabilities are fetched for the WHOLE sample in one query (the
per-oid reader in db.py would be N round-trips).

A mismatch is only meaningful when multisurvey_ztf holds the object's COMPLETE
light curve, so every statistic is also reported split on `lc_complete`
(|ms.lastmjd - alerce.lastmjd| <= --lc-tol days).

    MODEL_PATH=<local BHRF 2.1.0 pickle> python \
        feature_step/scripts/offline_compare_probabilities_batch.py \
        --xmatch-url http://127.0.0.1:8081
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper",
          PIPE / "libs" / "apf", PIPE / "libs" / "xmatch_client", PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sqlalchemy import text

from features.offline import db, xmatch
from features.offline.classify import classify_oid, load_squidward_model
from features.offline.probability_compare import (
    BHRF_CLASSIFIER_NAMES,
    compare_probability_frames,
    offline_dto_to_series,
)

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")
DEFAULT_SAMPLE = str(PIPE / "feature_step" / "features" / "offline" / "oids" / "compare_sample.json")


def _short(cname):
    """Trim the shared classifier prefix so head names fit a table column."""
    s = cname.replace("lc_classifier_BHRF_forced_phot", "")
    return s.lstrip("_") or "flat"


def fetch_stored_probabilities_batch(credentials, ztf_oids, classifier_names, version):
    """Stored BHRF probabilities for MANY oids in one query.

    Same filters as db.fetch_stored_probabilities (which is per-oid): the
    `classifier_name = ANY(...)` prunes to the 5 BHRF partitions and the
    per-partition PK makes each oid a cheap index lookup.
    """
    engine = db._make_engine(credentials)
    query = text(f"""
        SELECT oid, classifier_name, class_name, probability, ranking
        FROM {db.ALERCE_SCHEMA}.probability
        WHERE oid = ANY(:oids)
          AND classifier_version = :version
          AND classifier_name = ANY(:names)
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(
            query, conn,
            params={"oids": list(ztf_oids), "version": version,
                    "names": list(classifier_names)},
        )


def _verify_model(model):
    """Guard against the stale SNIbc pickle cached under /tmp (see OFFLINE_VS_LEGACY_VALIDATION.md)."""
    inner = getattr(model, "model", None)
    raw = getattr(inner, "list_of_classes", None)
    if raw is None:
        raw = getattr(model, "list_of_classes", [])
    classes = [str(c) for c in raw]
    if "SESN" not in classes or "SNIbc" in classes:
        sys.exit(f"error: wrong model loaded — expected SESN and no SNIbc, got {classes}")
    return classes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default=DEFAULT_SAMPLE, help="sample JSON from the sampler")
    ap.add_argument("--limit", type=int, default=None, help="only the first N sample rows")
    ap.add_argument("--version", default="2.1.0", help="stored classifier_version")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--lc-tol", type=float, default=1.0,
                    help="days of |ms.lastmjd - alerce.lastmjd| still counted as complete")
    ap.add_argument("--min-det", type=int, default=1)
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--model-path", default=os.getenv("MODEL_PATH"),
                    help="local BHRF 2.1.0 pickle (a URL is refused: it can silently "
                         "reuse a stale cached pickle)")
    ap.add_argument("--xmatch-url", default=os.getenv("XMATCH_URL", xmatch.DEFAULT_XMATCH_URL),
                    dest="xmatch_url",
                    help="Xwave URL for the live AllWISE crossmatch; '' uses the "
                         "(ZTF-empty) DB read, which nulls the WISE features")
    ap.add_argument("--json-out", default=None, help="write per-oid results here")
    args = ap.parse_args()

    if not args.model_path:
        sys.exit("error: --model-path (or MODEL_PATH) is required")
    if args.model_path.startswith(("http://", "https://")):
        sys.exit("error: --model-path must be a LOCAL pickle; a URL can silently reuse "
                 "a stale cached model (see OFFLINE_VS_LEGACY_VALIDATION.md)")
    os.environ["MODEL_PATH"] = str(Path(args.model_path).expanduser())

    sample = pd.read_json(args.sample)
    if args.limit:
        sample = sample.head(args.limit)
    print(f"sample: {len(sample)} oids from {args.sample}")

    model, name, model_version = load_squidward_model()
    classes = _verify_model(model)
    print(f"model: {name} version={model_version} ({len(classes)} flat classes, SESN ok)")
    if model_version != args.version:
        print(f"  WARNING: model version {model_version!r} != stored version {args.version!r}")

    # --- stored side: one query for the whole sample ---
    t0 = time.perf_counter()
    stored_all = fetch_stored_probabilities_batch(
        args.credentials, sample["ztf_oid"].tolist(), BHRF_CLASSIFIER_NAMES, args.version)
    print(f"stored: {len(stored_all)} rows for "
          f"{stored_all['oid'].nunique()}/{len(sample)} oids "
          f"({time.perf_counter() - t0:.1f}s)")
    heads_per_oid = stored_all.groupby("oid")["classifier_name"].nunique()
    print(f"  heads present per oid: min={heads_per_oid.min()} max={heads_per_oid.max()} "
          f"(all 5 for {(heads_per_oid == 5).sum()} oids)")
    stored_by_oid = dict(tuple(stored_all.groupby("oid")))

    # --- offline side: build extractor once, then loop ---
    from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
    from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
    preproc = ZTFLightcurvePreprocessor(drop_bogus=True)
    extractor = ZTFFeatureExtractor()
    print(f"xmatch: {'live Xwave @ ' + args.xmatch_url if args.xmatch_url else 'DB read (WISE -> NaN)'}\n")

    rows, all_diffs, skipped = [], [], []
    t_start = time.perf_counter()
    for i, r in enumerate(sample.itertuples(), 1):
        oid, ztf_oid = int(r.oid), r.ztf_oid
        stored = stored_by_oid.get(ztf_oid)
        if stored is None or stored.empty:
            skipped.append((ztf_oid, "no stored probabilities"))
            continue
        try:
            dto = classify_oid(oid, args.credentials, model, min_detections=args.min_det,
                               preprocessor=preproc, extractor=extractor,
                               xmatch_url=args.xmatch_url or None)
        except Exception as exc:  # noqa: BLE001 - one bad oid must not kill the run
            skipped.append((ztf_oid, f"classify raised: {type(exc).__name__}: {exc}"))
            continue
        if dto is None or dto.probabilities is None or len(dto.probabilities) == 0:
            skipped.append((ztf_oid, "offline produced no probabilities"))
            continue

        offline_by_name = offline_dto_to_series(dto)
        merged, summary = compare_probability_frames(
            offline_by_name, stored, rtol=args.rtol, atol=args.atol)

        gap = (abs(r.lastmjd - r.alerce_lastmjd)
               if pd.notna(getattr(r, "alerce_lastmjd", None)) else np.nan)
        rec = {
            "oid": oid, "ztf_oid": ztf_oid, "n_det": int(r.n_det),
            "stratum": r.stratum, "mjd_gap": float(gap) if pd.notna(gap) else None,
            "lc_complete": bool(pd.notna(gap) and gap <= args.lc_tol),
            "n_compared": summary["n_compared"], "match": summary["match"],
            "differ": summary["differ"],
            "only_offline": summary["only_offline"], "only_stored": summary["only_stored"],
            "rank1_agree": summary["rank1_agree"], "rank1_total": summary["rank1_total"],
            "passed": summary["passed"],
            "rank1": {_short(k): v for k, v in summary["rank1"].items()},
        }
        rows.append(rec)
        both = merged["prob_offline"].notna() & merged["prob_stored"].notna()
        all_diffs.extend(merged.loc[both, "abs_diff"].tolist())
        if i % 10 == 0:
            print(f"  {i}/{len(sample)} done ({time.perf_counter() - t_start:.0f}s)")

    if not rows:
        sys.exit("error: no oid produced a comparison")
    res = pd.DataFrame(rows)
    print(f"\ncompared {len(res)} oids in {time.perf_counter() - t_start:.0f}s "
          f"({len(skipped)} skipped)")

    # ---------------- report ----------------
    def block(title, sub):
        if sub.empty:
            return
        print(f"\n{'=' * 72}\n  {title}  (n={len(sub)})\n{'=' * 72}")
        print(f"  {'head':<12} {'rank-1 agree':>14} {'rate':>8}")
        for head in [_short(c) for c in BHRF_CLASSIFIER_NAMES]:
            vals = [row.get(head) for row in sub["rank1"] if row.get(head)]
            emitted = [v for v in vals if v["offline"] is not None and v["stored"] is not None]
            if not emitted:
                print(f"  {head:<12} {'-- head not emitted --':>14}")
                continue
            agree = sum(1 for v in emitted if v["agree"])
            print(f"  {head:<12} {f'{agree}/{len(emitted)}':>14} "
                  f"{100 * agree / len(emitted):>7.1f}%")
        full = int((sub["rank1_agree"] == sub["rank1_total"]).sum())
        print(f"\n  all emitted heads agree : {full}/{len(sub)} "
              f"({100 * full / len(sub):.1f}%)")
        print(f"  exact pass (probs too)  : {int(sub['passed'].sum())}/{len(sub)} "
              f"({100 * sub['passed'].sum() / len(sub):.1f}%)")
        cmp_n, mat = int(sub["n_compared"].sum()), int(sub["match"].sum())
        if cmp_n:
            print(f"  per-class probabilities : {mat}/{cmp_n} within "
                  f"rtol={args.rtol} atol={args.atol} ({100 * mat / cmp_n:.1f}%)")

    block("ALL OIDS", res)
    block(f"LC COMPLETE (mjd gap <= {args.lc_tol}d)", res[res["lc_complete"]])
    block(f"LC INCOMPLETE (mjd gap > {args.lc_tol}d)", res[~res["lc_complete"]])

    if all_diffs:
        d = np.array(all_diffs)
        print(f"\n  per-class |prob_offline - prob_stored| over {len(d)} class rows:")
        for q in (50, 90, 99):
            print(f"    p{q}: {np.percentile(d, q):.4f}")
        print(f"    max: {d.max():.4f}   mean: {d.mean():.4f}")

    print("\n  rank-1 (flat head) agreement by n_det stratum:")
    res["flat_agree"] = [r.get("flat", {}).get("agree", False) for r in res["rank1"]]
    by = res.groupby("stratum").agg(n=("oid", "size"), agree=("flat_agree", "sum"))
    by["rate"] = (100 * by["agree"] / by["n"]).round(1)
    print(by.to_string())

    dis = res[~res["flat_agree"]]
    if not dis.empty:
        print(f"\n  {len(dis)} oids where the flat rank-1 class DISAGREES:")
        print(f"    {'ztf_oid':<14} {'n_det':>6} {'gap_d':>7}  {'offline':<14} {'stored':<14}")
        for r in dis.itertuples():
            f = r.rank1.get("flat", {})
            gap = f"{r.mjd_gap:.1f}" if r.mjd_gap is not None else "n/a"
            print(f"    {r.ztf_oid:<14} {r.n_det:>6} {gap:>7}  "
                  f"{str(f.get('offline')):<14} {str(f.get('stored')):<14}")

    if skipped:
        print(f"\n  skipped ({len(skipped)}):")
        for z, why in skipped[:20]:
            print(f"    {z}: {why}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
