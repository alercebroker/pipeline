#!/usr/bin/env python
"""Prove the batched runner reproduces the validated single-oid path exactly.

offline_run_batch.py replaces `classify.classify_oid`'s per-oid reads with one
batched read per minibatch. That is only a performance change if the inputs it
hands the extractor are byte-identical to what the single-oid path hands it --
a grouping bug (wrong oid's reference rows, a dropped PS1 row, an AllWISE match
attached to the wrong object) would silently shift the predicted class, exactly
the failure mode the 100-oid comparison was built to detect.

So: run both paths over the same oids and diff the flat probability vectors.

    MODEL_PATH=... python feature_step/scripts/offline_verify_batch_equivalence.py \
        --n 12 --min-n-det 20 --xmatch-url ''
"""
import argparse
import os
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE / "feature_step" / "scripts"))

import offline_run_batch as R          # sets the thread pins + sys.path

import numpy as np
import pandas as pd

from features.offline import db, xmatch
from features.offline.classify import classify_oid, load_squidward_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--min-n-det", type=int, default=20)
    ap.add_argument("--minibatch", type=int, default=6)
    ap.add_argument("--credentials", default=R.DEFAULT_CREDENTIALS)
    ap.add_argument("--xmatch-url", default=os.getenv("XMATCH_URL", ""), dest="xmatch_url")
    args = ap.parse_args()

    oids = [int(o) for o in R.select_oids(args.credentials, args.min_n_det, args.n)]
    print(f"{len(oids)} oids, minibatch={args.minibatch}, "
          f"xmatch={'live ' + args.xmatch_url if args.xmatch_url else 'DB read (WISE NaN)'}")

    cfg = {"credentials": args.credentials, "schema": db.SCHEMA,
           "xmatch_url": args.xmatch_url or None, "out_dir": "/tmp",
           "minibatch": args.minibatch, "min_detections": 1,
           "features": True, "retries": 2, "warnings": False}
    R._MODEL, _, _ = load_squidward_model()
    R._init_worker(cfg)
    model = R._W["model"]

    # --- batched path ---
    batched = {}
    for s in range(0, len(oids), args.minibatch):
        mb = oids[s:s + args.minibatch]
        for oid, (msg, refs, allwise) in R.fetch_minibatch(mb, cfg).items():
            ao = R.compute_astro_object(msg, refs, allwise, 1,
                                        preprocessor=R._W["preprocessor"],
                                        extractor=R._W["extractor"])
            dto = R.classify_astro_object(ao, msg, model)
            batched[oid] = dto.probabilities.iloc[0]

    # --- reference path (one oid at a time, as validated) ---
    def reference_pass():
        out = {}
        for oid in oids:
            dto = classify_oid(oid, args.credentials, model, min_detections=1,
                               preprocessor=R._W["preprocessor"],
                               extractor=R._W["extractor"],
                               xmatch_url=args.xmatch_url or None)
            if dto is not None and len(dto.probabilities):
                out[oid] = dto.probabilities.iloc[0]
        return out

    # TWO passes of the SAME single-oid path. This is the control: several
    # extractors (SPM, ulens, TDE, the periodogram) fit with stochastic
    # initialisation, so the single-oid path does not necessarily reproduce
    # ITSELF bit for bit. Without this baseline, any batched-vs-single delta
    # looks like a batching bug when it may just be the extractor's own noise.
    reference = reference_pass()
    control = reference_pass()

    def delta(a, b, oid):
        x, y = a[oid], b[oid]
        return float(np.nanmax(np.abs(x.reindex(y.index).to_numpy() - y.to_numpy())))

    shared = sorted(set(batched) & set(reference) & set(control))
    only = (set(batched) ^ set(reference)) | (set(reference) ^ set(control))

    print(f"\n{'oid':>20} {'class':<14} {'batch-vs-single':>16} {'single-vs-single':>18}  ok")
    n_same = n_within = 0
    for oid in shared:
        d_batch = delta(batched, reference, oid)
        d_self = delta(control, reference, oid)
        same = batched[oid].idxmax() == reference[oid].idxmax()
        # The bar: batching must not perturb the result by MORE than the
        # extractor perturbs itself between two identical runs.
        within = d_batch <= max(d_self, 1e-12)
        n_same += same
        n_within += within
        print(f"{oid:>20} {reference[oid].idxmax():<14} {d_batch:>16.3e} "
              f"{d_self:>18.3e}  {'OK' if same and within else 'XX'}")

    n = len(shared)
    print(f"\n  compared                        : {n}")
    print(f"  same rank-1 class               : {n_same}/{n}")
    print(f"  batched delta <= self delta     : {n_within}/{n}")
    if only:
        print(f"  ONLY IN ONE PATH                : {sorted(only)}   <-- grouping bug")

    if all(delta(control, reference, o) == 0.0 for o in shared):
        print("\n  (the single-oid path IS deterministic here, so any nonzero\n"
              "   batch-vs-single delta is a real batching difference)")
    else:
        print("\n  NOTE: the single-oid path does not reproduce itself either --\n"
              "  the extractors fit stochastically. Bitwise equality is not the\n"
              "  right bar; same class + delta within self-noise is.")

    ok = n > 0 and not only and n_same == n and n_within == n
    print("\nRESULT:", "PASS - batching does not perturb the result beyond the "
          "extractor's own run-to-run noise." if ok else
          "FAIL - batching changes the result by more than the extractor's own noise.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
