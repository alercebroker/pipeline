#!/usr/bin/env python
"""Generate the offline ZTF feature_name_lut + feature_version_lut fixture.

Runs the real extractor on one (or more) representative oid(s), collects the
FULL set of band-less feature names (NOT NaN-filtered — we want the complete
feature schema, not one object's non-NaN subset), in EXTRACTOR (natural) order
— the order features come out of the composite, NOT alphabetical — and assigns
ids 0..N-1. Prints a ready-to-paste Python literal for
feature_step/features/offline/feature_lut.py.

    conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_generate_feature_lut.py --oid 36028941624528297
"""
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]
for p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper"):
    sys.path.insert(0, str(p))

import argparse
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from features.offline import db, lc_features, feature_lut
from features.offline.message import build_message

# Mirror the back-compat name fixes in prepare_ao_features_for_db.
_NAME_FIXES = {
    "Power_rate_1_4": "Power_rate_1/4",
    "Power_rate_1_3": "Power_rate_1/3",
    "Power_rate_1_2": "Power_rate_1/2",
}
DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")


def collect(credentials, oids):
    # Preserve extractor (natural) emission order: first occurrence across the
    # concatenated ao.features, deduped. NOT sorted alphabetically — the id order
    # must mirror the order the composite emits features.
    names = []
    seen = set()
    for oid in oids:
        dets = db.fetch_detections(credentials, [oid])
        forced = db.fetch_forced_photometry(credentials, [oid])
        ps1 = db.fetch_ps1(credentials, [oid])
        allwise = db.fetch_allwise(credentials, [oid])
        refs = db.fetch_references(credentials, [oid])
        message = build_message(oid, dets, forced, ps1)
        ao = lc_features.compute_astro_object(message, refs, allwise)
        if ao is None:
            print(f"# oid {oid}: too few detections, skipped", file=sys.stderr)
            continue
        feats = ao.features  # NOT NaN-filtered
        for name in feats["name"].replace(_NAME_FIXES):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser(description="Generate offline ZTF feature LUT fixture.")
    ap.add_argument("--oid", type=int, action="append", required=True,
                    help="Multisurvey bigint oid (repeat to union name sets).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    args = ap.parse_args()

    names = collect(args.credentials, args.oid)
    try:
        versions = [_pkg_version("feature-step")]
    except PackageNotFoundError:
        # feature-step not pip-installed (running from source); fall back to the
        # canonical offline version, same as offline_compute_features.
        versions = [feature_lut.default_version_name()]
    print(f"# {len(names)} feature names; versions={versions}")
    print("FEATURE_NAME_LUT = {")
    for i, n in enumerate(names):
        print(f"    {i}: {n!r},")
    print("}")
    print("FEATURE_VERSION_LUT = {")
    for i, v in enumerate(versions):
        print(f"    {i}: {v!r},")
    print("}")


if __name__ == "__main__":
    main()
