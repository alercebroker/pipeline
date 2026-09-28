#!/usr/bin/env python
"""Where does a unit's time go RIGHT NOW? One minibatch, one worker, a stopwatch
around each phase.

A worker's unit is ten rounds of: four SQL reads, one Xwave call, then feature
computation + classification for 500 objects. top shows the CPU phase; the
database sees only the SQL phase; nothing sees the Xwave call or attributes the
total. When the same objects cost 2x what they cost in August, this says which
phase grew. Run it on the server WHILE the run is going, against a unit the run
has not reached, and compare with the reference run's seconds/oid on the same
oid range (printed alongside if --ref-dir is given).

    poetry run python scripts/offline_time_minibatch.py \
        --oid-file $RUN/oids/reprocesados.npy --unit 400 --ref-dir $RUN/bhrf_run

Reads only. No database writes, no shards, no manifest. It does load the model
and build the extractor, so it costs one worker's memory for a few minutes.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_run_batch as M  # noqa: E402  (brings the pipeline paths with it)
from features.offline import db, xmatch  # noqa: E402
from features.offline.message import build_message  # noqa: E402


def reference_cost(ref_dir: Path, lo: int, hi: int):
    secs = oids = 0.0
    for p in (ref_dir / "manifests").glob("unit_*.json"):
        m = json.loads(p.read_text())
        if m["oid_hi"] < lo or m["oid_lo"] > hi:
            continue
        secs += m["elapsed_s"]
        oids += m["n_oids"]
    return secs / oids if oids else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oid-file", required=True)
    ap.add_argument("--unit", type=int, required=True, help="unit index into the array")
    ap.add_argument("--minibatch-index", type=int, default=0,
                    help="which of the unit's minibatches (default: the first)")
    ap.add_argument("--unit-size", type=int, default=5000)
    ap.add_argument("--minibatch", type=int, default=xmatch.DEFAULT_BATCH_SIZE)
    ap.add_argument("--credentials", default=M.DEFAULT_CREDENTIALS)
    ap.add_argument("--schema", default=db.SCHEMA)
    ap.add_argument("--xmatch-url", default=xmatch.DEFAULT_XMATCH_URL)
    ap.add_argument("--min-detections", type=int, default=1)
    ap.add_argument("--ref-dir", type=Path, help="a finished run: prints its s/oid on this range")
    ap.add_argument("--n-oids", type=int, default=0,
                    help="only the first N oids of the minibatch (default: all)")
    args = ap.parse_args()

    arr = np.load(args.oid_file)
    lo = args.unit * args.unit_size + args.minibatch_index * args.minibatch
    oids = [int(o) for o in arr[lo:lo + args.minibatch]]
    if args.n_oids:
        oids = oids[:args.n_oids]
    print(f"unit {args.unit} minibatch {args.minibatch_index}: {len(oids)} oids, "
          f"{oids[0]} .. {oids[-1]}")

    cfg = {"credentials": args.credentials, "schema": args.schema, "load_db": False,
           "write_credentials": None, "no_shards": True, "xmatch_url": args.xmatch_url,
           "out_dir": "/nonexistent", "minibatch": args.minibatch,
           "min_detections": args.min_detections, "features": True, "retries": 1,
           "warnings": False}
    cfg["feature_lut"] = db.fetch_feature_name_lut(args.credentials, schema=args.schema)
    cfg["feature_version_id"] = db.fetch_feature_version_id(
        args.credentials, M.default_version_name(), schema=args.schema)

    t = time.perf_counter()
    M._MODEL = M.load_squidward_model()[0]
    M._init_worker(cfg)
    print(f"setup (model + extractor + taxonomy): {time.perf_counter() - t:7.1f}s  (paid once per worker, not per unit)")

    phases = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        out = fn()
        phases[name] = time.perf_counter() - t0
        return out

    dets = timed("sql detections", lambda: db.fetch_detections(args.credentials, oids))
    forced = timed("sql forced_photometry", lambda: db.fetch_forced_photometry(args.credentials, oids))
    ps1 = timed("sql ps1", lambda: db.fetch_ps1(args.credentials, oids))
    refs = timed("sql references", lambda: db.fetch_references(args.credentials, oids))

    dets_by, dets_empty = M._by_oid(dets)
    forced_by, forced_empty = M._by_oid(forced)
    ps1_by, ps1_empty = M._by_oid(ps1)
    refs_by, refs_empty = M._by_oid(refs)

    messages = {}
    for oid in oids:
        d = dets_by.get(oid)
        if d is None or len(d) == 0:
            continue
        messages[oid] = build_message(oid, d, forced_by.get(oid, forced_empty),
                                      ps1_by.get(oid, ps1_empty))

    n_det = [len(dets_by[o]) for o in messages]
    n_forced = [len(forced_by.get(o, forced_empty)) for o in messages]
    print(f"\ninputs: {len(messages)} oids with detections of {len(oids)}")
    print(f"  detections/oid      : mean {statistics.mean(n_det):7.1f}  median {statistics.median(n_det):6.0f}  max {max(n_det):6d}  total {sum(n_det):8d}")
    print(f"  forced phot/oid     : mean {statistics.mean(n_forced):7.1f}  median {statistics.median(n_forced):6.0f}  max {max(n_forced):6d}  total {sum(n_forced):8d}")

    mb_oids = list(messages)
    matches = timed("xwave crossmatch", lambda: xmatch.compute_matches(
        mb_oids, [messages[o]["meanra"] for o in mb_oids],
        [messages[o]["meandec"] for o in mb_oids], base_url=args.xmatch_url))
    allwise_by, _ = M._by_oid(xmatch.matches_to_allwise_df(matches))
    allwise_empty = allwise_by[next(iter(allwise_by))].iloc[0:0] if allwise_by else None
    n_no_allwise = sum(1 for o in mb_oids if o not in allwise_by)
    print(f"  no AllWISE          : {n_no_allwise} of {len(mb_oids)} ({n_no_allwise / max(1, len(mb_oids)):.1%})")

    per_oid = []
    n_ok = n_unclass = n_err = 0
    t0 = time.perf_counter()
    for oid in mb_oids:
        aw = allwise_by.get(oid)
        if aw is None:
            import pandas as pd
            aw = pd.DataFrame(columns=["oid", "W1", "W2", "W3", "W4"])
        t1 = time.perf_counter()
        try:
            p_rows, f_rows = M.process_oid(oid, messages[oid], refs_by.get(oid, refs_empty), aw, cfg)
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            print(f"  error on {oid}: {type(exc).__name__}: {exc}")
            continue
        per_oid.append(time.perf_counter() - t1)
        if p_rows:
            n_ok += 1
        else:
            n_unclass += 1
    phases["compute (features + classify)"] = time.perf_counter() - t0

    total = sum(phases.values())
    print(f"\nphases for this minibatch ({len(oids)} oids):")
    for k, v in phases.items():
        print(f"  {k:32s} {v:8.1f}s  {v / total:5.1%}  {v / len(oids):6.3f} s/oid")
    print(f"  {'TOTAL':32s} {total:8.1f}s         {total / len(oids):6.3f} s/oid")
    if per_oid:
        per_oid.sort()
        print(f"\ncompute per oid: median {statistics.median(per_oid):.3f}s  p90 {per_oid[int(0.9 * len(per_oid)) - 1]:.3f}s  max {per_oid[-1]:.3f}s"
              f"   (classified {n_ok}, unclassifiable {n_unclass}, errors {n_err})")
    if args.ref_dir:
        r = reference_cost(args.ref_dir, oids[0], oids[-1])
        if r is None:
            print(f"\nreference {args.ref_dir}: no unit covers this oid range")
        else:
            print(f"\nreference {args.ref_dir}: {r:.3f} s/oid on this oid range  ->  now/ref = {total / len(oids) / r:.2f}")
            print("  (the reference number includes its own DB write at the end of the unit; this one has none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
