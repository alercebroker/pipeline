#!/usr/bin/env python
"""Project a full run from the manifests a probe left behind.

Every finished unit writes manifests/unit_NNNNNNN.json with how many oids it
processed, how long it took, how many rows it produced and the worker's peak
RSS. That is a measurement from the target machine, which is worth more than any
arithmetic done elsewhere -- so this reads them and extrapolates to the whole
oid list instead of guessing.

    python scripts/offline_estimate.py /data/bhrf_probe1 \\
        --oid-file /data/oids/run.npy --workers 64

Read the tail numbers, not just the mean: a unit is one worker's task and its
oids run in series, so the slowest unit is what decides when the last worker
finishes, and the mean hides it.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path


def estimate(manifests, n_total_oids: int, workers: int) -> dict:
    """Manifests + catalogue size + worker count -> projected run.

    Timing is summed as CORE-seconds: elapsed_s is one unit inside one worker,
    so the total wall clock is core-seconds / workers. Rows and the no-AllWISE
    rate are per-oid rates measured on the probe, applied to the whole list.
    """
    if not manifests:
        raise ValueError("no manifests to estimate from -- did the probe finish a unit?")

    oids = sum(m["n_oids"] for m in manifests)
    core_s = sum(m["elapsed_s"] for m in manifests)
    per_unit = sorted(m["elapsed_s"] for m in manifests)
    core_s_per_oid = core_s / oids

    # Present only when the probe ran a version that records it; inventing a
    # number here would be worse than saying nothing, since memory is the risk
    # this whole projection exists to size.
    rss = [m["peak_rss_mb"] for m in manifests if m.get("peak_rss_mb")]
    peak = max(rss) if rss else None

    def _rate(field):
        return sum(m.get(field, 0) for m in manifests) / oids

    return {
        "units": len(manifests),
        "oids_measured": oids,
        "core_s_per_oid": core_s_per_oid,
        "hours": core_s_per_oid * n_total_oids / workers / 3600,
        "unit_s_p50": statistics.median(per_unit),
        "unit_s_p90": per_unit[min(len(per_unit) - 1, int(0.9 * len(per_unit)))],
        "unit_s_max": per_unit[-1],
        "prob_rows": round(_rate("prob_rows") * n_total_oids),
        "feat_rows": round(_rate("feat_rows") * n_total_oids),
        "no_allwise_rate": _rate("n_no_allwise"),
        "peak_rss_mb": peak,
        "projected_rss_gb": peak * workers / 1024 if peak else None,
    }


def load_manifests(out_dir: Path) -> list:
    return [json.loads(p.read_text())
            for p in sorted((out_dir / "manifests").glob("unit_*.json"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="a finished (or partial) run directory.")
    ap.add_argument("--oid-file", type=Path,
                    help=".npy/.txt oid list the real run will use; sets the total.")
    ap.add_argument("--n-total", type=int,
                    help="catalogue size, if you would rather not load the oid file.")
    ap.add_argument("--workers", type=int, required=True,
                    help="worker count the REAL run will use (not the probe's).")
    args = ap.parse_args()

    manifests = load_manifests(args.out_dir)
    if not manifests:
        sys.exit(f"no manifests under {args.out_dir}/manifests -- nothing to estimate from")

    if args.n_total:
        n_total = args.n_total
    elif args.oid_file:
        import numpy as np
        n_total = len(np.load(args.oid_file, mmap_mode="r")
                      if args.oid_file.suffix == ".npy"
                      else np.loadtxt(args.oid_file, dtype="int64", ndmin=1))
    else:
        ap.error("pass --oid-file or --n-total")

    e = estimate(manifests, n_total, args.workers)
    gb = lambda n: n / 1e9

    print(f"\nmeasured: {e['units']} units, {e['oids_measured']:,} oids "
          f"in {args.out_dir}")
    print(f"  per oid       : {e['core_s_per_oid']:.3f} core-s")
    print(f"  per unit      : p50 {e['unit_s_p50']:.0f}s  p90 {e['unit_s_p90']:.0f}s  "
          f"max {e['unit_s_max']:.0f}s")
    print(f"  no AllWISE    : {e['no_allwise_rate']:.1%}  (~14% expected)")
    if e["peak_rss_mb"]:
        print(f"  peak RSS/worker: {e['peak_rss_mb']:,.0f} MB")

    print(f"\nprojected: {n_total:,} oids on {args.workers} workers")
    print(f"  elapsed       : {e['hours']:.1f} h  ({e['hours']/24:.1f} days)")
    print(f"  probability   : {gb(e['prob_rows']):.2f}e9 rows")
    print(f"  feature       : {gb(e['feat_rows']):.2f}e9 rows")
    if e["projected_rss_gb"]:
        print(f"  RSS all workers: {e['projected_rss_gb']:,.0f} GB "
              "<- compare against the host's RAM before scaling up")
    else:
        print("  RSS: not recorded by this probe (older manifests); rerun to get it")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
