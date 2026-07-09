#!/usr/bin/env python
"""Minimal: run the Xwave crossmatch client on one oid and print what it returns.

Mirrors the feature step / offline path: DB -> build_message (meanra/meandec) ->
XmatchClient.conesearch_with_metadata at that cone center. Defaults to the
canonical lc-classification test oid (ZTF17aaabauy = 36028933559755080).

    conda run --no-capture-output -n hbrf_rubin python \
        feature_step/scripts/offline_xmatch_oid.py

    # all catalogs (see gaia too), then narrow radius:
    feature_step/scripts/offline_xmatch_oid.py --catalogs all --radius 1.5
"""
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier",
          PIPE / "libs" / "idmapper", PIPE / "libs" / "apf",
          PIPE / "libs" / "xmatch_client", PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(p))

import argparse
import os

from features.offline import db, xmatch
from features.offline.message import build_message

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")
DEFAULT_OID = 36028933559755080  # ZTF17aaabauy


def main():
    ap = argparse.ArgumentParser(description="Run the Xwave xmatch client on one oid.")
    ap.add_argument("--oid", type=int, default=DEFAULT_OID,
                    help="Multisurvey bigint oid (default: %(default)s = ZTF17aaabauy).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS,
                    help="Path to DB credentials JSON (read-only).")
    ap.add_argument("--xmatch-url", default=os.getenv("XMATCH_URL", xmatch.DEFAULT_XMATCH_URL),
                    dest="xmatch_url", help="Xwave service URL (default: %(default)s).")
    ap.add_argument("--radius", type=float, default=xmatch.DEFAULT_RADIUS,
                    help="Cone radius in arcsec (default: %(default)s).")
    ap.add_argument("--catalogs", default="allwise",
                    help="Comma-separated catalogs to keep, or 'all' for every catalog "
                         "the client returns (default: allwise).")
    ap.add_argument("--nneighbor", type=int, default=1,
                    help="Neighbors per position. The XmatchClient hardcodes 1 (single "
                         "GLOBAL nearest across all catalogs). >1 bypasses the client and "
                         "hits v1/bulk-conesearch raw so each catalog can contribute "
                         "(default: 1).")
    args = ap.parse_args()

    catalogs = None if args.catalogs.lower() == "all" else args.catalogs.split(",")

    # DB -> meanra/meandec (same cone center the step/offline path uses).
    oids = [args.oid]
    dets = db.fetch_detections(args.credentials, oids)
    forced = db.fetch_forced_photometry(args.credentials, oids)
    ps1 = db.fetch_ps1(args.credentials, oids)
    message = build_message(args.oid, dets, forced, ps1)
    ra, dec = message["meanra"], message["meandec"]

    print("=" * 70)
    print(f"  oid    : {args.oid}")
    print(f"  center : ra={ra:.6f}  dec={dec:.6f}  (n_det={len(dets)})")
    print(f"  query  : url={args.xmatch_url}  radius={args.radius}\"  "
          f"catalogs={catalogs or 'all'}  nneighbor={args.nneighbor}")
    print("=" * 70)

    if args.nneighbor != 1:
        # The client hardcodes nneighbor=1; hit the service raw to let each
        # catalog contribute its own nearest neighbor(s).
        import requests
        resp = requests.post(
            f"{args.xmatch_url.rstrip('/')}/v1/bulk-conesearch",
            json={"oids": [str(args.oid)], "ra": [ra], "dec": [dec],
                  "radius": args.radius, "nneighbor": args.nneighbor},
        )
        resp.raise_for_status()
        results = resp.json()
        total = 0
        for result in results:
            for cat in result.get("Data", []):
                name = cat["catalog"]
                if catalogs and name not in catalogs:
                    continue
                for row in cat.get("data", []):
                    total += 1
                    print(f"  catalog={name:12s} id={row.get('id') or row.get('match_id')} "
                          f"dist={row.get('distance')}")
        print(f"\n{total} match(es) within {args.radius}\".")
        return

    matches = xmatch.compute_matches(
        oids, [ra], [dec], base_url=args.xmatch_url,
        radius=args.radius, catalogs=catalogs)

    if not matches:
        print("\nNo matches returned.")
        return

    print(f"\n{len(matches)} match(es):")
    for m in matches:
        print(f"  catalog={m.get('catalog'):12s} match_id={m.get('match_id')} "
              f"dist={m.get('distance')}")
        meta = m.get("metadata") or {}
        if meta:
            print(f"      metadata keys: {sorted(meta)}")


if __name__ == "__main__":
    main()
