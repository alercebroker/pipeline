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
                    help="Neighbors per position. The XmatchClient issues one "
                         "per-catalog request (nneighbor=1 => the nearest match in "
                         "each catalog). >1 bypasses the client and hits "
                         "v1/bulk-conesearch raw for extra neighbors (default: 1).")
    ap.add_argument("--save", action="store_true",
                    help="Upsert the computed crossmatch into <schema>.xmatch "
                         "(dry-run unless --execute). No model needed.")
    ap.add_argument("--execute", action="store_true",
                    help="With --save, actually write. Requires --write-credentials.")
    ap.add_argument("--write-credentials", default=None, dest="write_credentials",
                    help="Write-capable DB credentials JSON (default credentials are read-only).")
    args = ap.parse_args()

    if args.execute and not args.save:
        ap.error("--execute requires --save (nothing to write otherwise)")
    if args.execute and not args.write_credentials:
        ap.error("--execute requires --write-credentials (the default credentials are read-only)")
    if args.save and args.nneighbor != 1:
        ap.error("--save works with the client path only (nneighbor=1); the raw "
                 "multi-neighbor path is diagnostic-only")

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
        # The client only asks for nneighbor=1 per catalog; hit the service raw
        # to fetch more than one neighbor per catalog.
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

    if args.save:
        outcome = xmatch.persist_matches(
            matches, write_credentials=args.write_credentials or args.credentials,
            execute=args.execute)
        print(f"\nsave: {outcome} -> {db.SCHEMA}.xmatch")
        if not args.execute:
            for r in xmatch.build_xmatch_rows(matches):
                print(f"  would write: {r}")
            print("  (dry-run — pass --execute with --write-credentials to write)")


if __name__ == "__main__":
    main()
