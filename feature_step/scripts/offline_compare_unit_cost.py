#!/usr/bin/env python
"""Is this run slower than a finished run was on the SAME objects?

A run that decays can be the database saturating under the workers, or just the
region of the sorted oid array it is in: the ZTF17 objects at the front carry
the longest light curves and their units take hours everywhere. The two need
opposite fixes (fewer --workers vs a longer --stall-timeout), and elapsed_s
alone cannot tell them apart.

The finished run's manifests can. Every manifest records oid_lo, oid_hi, n_oids
and elapsed_s, so for each unit of the current run this finds the reference
run's units covering the same oid range and compares seconds per oid. Ratio
near 1: same objects, same cost -- the region is heavy, not the database.
Ratio climbing with unit index: the shared resource is degrading.

Reads manifests only. No DB, no network, stdlib only.

    python3 scripts/offline_compare_unit_cost.py $RUN/bhrf_reproc $RUN/bhrf_run
"""
import argparse
import json
import statistics
from pathlib import Path


def load(out_dir: Path) -> list:
    mans = [json.loads(p.read_text()) for p in sorted((out_dir / "manifests").glob("unit_*.json"))]
    return sorted(mans, key=lambda m: m["oid_lo"])


def reference_cost(ref: list, lo: int, hi: int) -> float | None:
    """Seconds per oid of the reference units overlapping [lo, hi], weighted by oids."""
    secs = oids = 0.0
    for m in ref:
        if m["oid_hi"] < lo or m["oid_lo"] > hi:
            continue
        secs += m["elapsed_s"]
        oids += m["n_oids"]
    return secs / oids if oids else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="the run in question (partial is fine)")
    ap.add_argument("ref_dir", type=Path, help="a finished run over the same catalogue")
    ap.add_argument("--last", type=int, default=0,
                    help="only the N most recently finished units (default: all)")
    args = ap.parse_args()

    cur = load(args.run_dir)
    ref = load(args.ref_dir)
    if args.last:
        cur = sorted(cur, key=lambda m: m["unit"])[-args.last:]
    print(f"{'unit':>6} {'oids':>6} {'s/oid now':>10} {'s/oid ref':>10} {'ratio':>6}")
    ratios = []
    for m in sorted(cur, key=lambda m: m["unit"]):
        now = m["elapsed_s"] / m["n_oids"]
        r = reference_cost(ref, m["oid_lo"], m["oid_hi"])
        if r is None:
            print(f"{m['unit']:>6} {m['n_oids']:>6} {now:>10.3f} {'--':>10} {'--':>6}")
            continue
        ratios.append((m["unit"], now / r))
        print(f"{m['unit']:>6} {m['n_oids']:>6} {now:>10.3f} {r:>10.3f} {now / r:>6.2f}")
    if ratios:
        vals = [x for _, x in ratios]
        first = [x for _, x in ratios[: max(1, len(ratios) // 4)]]
        last = [x for _, x in ratios[-max(1, len(ratios) // 4):]]
        print(f"\nratio now/ref: median {statistics.median(vals):.2f}  "
              f"first quarter {statistics.median(first):.2f}  "
              f"last quarter {statistics.median(last):.2f}")
        print("  ~1 and flat  -> same objects cost the same: heavy region, raise --stall-timeout")
        print("  >1 and rising -> the same objects got slower: shared resource saturating, lower --workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
