"""Benchmark + accuracy harness for the MHAOV MultiBandPeriodogram path.

Times the exact production sequence (set_data -> optimal_frequency_grid_evaluation
-> optimal_finetune_best_frequencies) across representative light curves, and
records both wall-time and result quality (recovered period, peak height) so the
performance refactor can be measured and the float64 accuracy gain documented.

Usage:
    python benchmarks/bench_mhaov.py --label baseline_f32
    python benchmarks/bench_mhaov.py --label loopmove_f32
    python benchmarks/bench_mhaov.py --label final_f64

Results for every label are stored in benchmarks/bench_results.json and rendered
into benchmarks/RESULTS.md (a table comparing all labels). Timings use a warmup
run followed by the median of N repeats.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from conftest import make_lightcurve  # noqa: E402

from P4J import MultiBandPeriodogram  # noqa: E402

HERE = os.path.dirname(__file__)
STORE = os.path.join(HERE, "bench_results.json")
RESULTS_MD = os.path.join(HERE, "RESULTS.md")

# (name, period, n_points, baseline, bands, smallest_period, largest_period)
SCENARIOS = [
    ("single_2yr_N500", 0.2734, 500, (59000.0, 61000.0), ("g",), 0.05, 5.0),
    ("single_2yr_N1000", 0.2734, 1000, (59000.0, 61000.0), ("g",), 0.05, 5.0),
    ("multi_2yr_N500x2", 0.2734, 500, (59000.0, 61000.0), ("g", "r"), 0.05, 5.0),
    ("short_10yr_N800", 0.0721, 800, (59000.0, 62650.0), ("g",), 0.03, 2.0),
]
REPEATS = 3


def run_once(lc, smallest_period, largest_period):
    mjd, mag, err, fid = lc
    p = MultiBandPeriodogram(method="MHAOV")
    p.set_data(mjd, mag, err, fid)
    p.optimal_frequency_grid_evaluation(
        smallest_period=smallest_period, largest_period=largest_period, shift=0.1
    )
    grid_size = p.freq.size
    p.optimal_finetune_best_frequencies(times_finer=10.0, n_local_optima=10)
    best_freq, best_per = p.get_best_frequencies()
    return grid_size, float(1.0 / best_freq[0]), float(best_per[0])


def benchmark():
    rows = []
    for name, period, n, baseline, bands, sp, lp in SCENARIOS:
        lc = make_lightcurve(period, n_points=n, baseline=baseline, seed=0, bands=bands)
        run_once(lc, sp, lp)  # warmup (build/compile caches)
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            grid_size, recovered, peak = run_once(lc, sp, lp)
            times.append(time.perf_counter() - t0)
        rows.append(
            {
                "scenario": name,
                "true_period": period,
                "grid_size": grid_size,
                "median_s": statistics.median(times),
                "recovered_period": recovered,
                "rel_err": abs(recovered - period) / period,
                "peak_height": peak,
            }
        )
        print(
            f"  {name:20s} grid={grid_size:>7d} t={statistics.median(times):7.3f}s "
            f"recov={recovered:.6f} relerr={rows[-1]['rel_err']:.2e} peak={peak:.1f}"
        )
    return rows


def render_md(store):
    labels = list(store.keys())
    scenarios = [s[0] for s in SCENARIOS]
    lines = ["# P4J MHAOV Benchmark Results", ""]
    lines.append(f"Env: python {sys.version.split()[0]}, numpy {np.__version__}. "
                 f"Median of {REPEATS} repeats (after warmup).")
    lines.append("")

    # Timing table
    lines.append("## Wall time (seconds)")
    lines.append("")
    lines.append("| scenario | grid | " + " | ".join(labels) + " | speedup(first→last) |")
    lines.append("|" + "---|" * (len(labels) + 3))
    for sc in scenarios:
        grid = next((r["grid_size"] for lab in labels for r in store[lab] if r["scenario"] == sc), "-")
        cells = []
        first = last = None
        for lab in labels:
            r = next((r for r in store[lab] if r["scenario"] == sc), None)
            if r:
                cells.append(f"{r['median_s']:.3f}")
                if first is None:
                    first = r["median_s"]
                last = r["median_s"]
            else:
                cells.append("-")
        speedup = f"{first / last:.1f}x" if first and last else "-"
        lines.append(f"| {sc} | {grid} | " + " | ".join(cells) + f" | {speedup} |")

    # Accuracy table
    lines.append("")
    lines.append("## Accuracy (recovered period rel. error / coarse peak height)")
    lines.append("")
    lines.append("| scenario | true P | " + " | ".join(labels) + " |")
    lines.append("|" + "---|" * (len(labels) + 2))
    for sc in scenarios:
        trueP = next((r["true_period"] for lab in labels for r in store[lab] if r["scenario"] == sc), "-")
        cells = []
        for lab in labels:
            r = next((r for r in store[lab] if r["scenario"] == sc), None)
            cells.append(f"{r['rel_err']:.1e} / {r['peak_height']:.0f}" if r else "-")
        lines.append(f"| {sc} | {trueP} | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="label for this run (e.g. baseline_f32)")
    args = ap.parse_args()

    print(f"benchmark label={args.label}")
    rows = benchmark()

    store = {}
    if os.path.exists(STORE):
        with open(STORE) as f:
            store = json.load(f)
    store[args.label] = rows
    with open(STORE, "w") as f:
        json.dump(store, f, indent=2)
    with open(RESULTS_MD, "w") as f:
        f.write(render_md(store))
    print(f"\nstored under '{args.label}'; wrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
