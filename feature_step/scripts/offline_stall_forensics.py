#!/usr/bin/env python
"""Post-mortem for an offline_run_batch.py run that hit the stall abort.

    ABORTING: stalled for 30.0 min with no progress.

That message says a stall happened; it cannot say what kind. This reads the
--out-dir and answers the question the runner's stdout cannot:

  * WHEN did progress stop? Every finished unit writes manifests/unit_*.json
    LAST -- after its shards and, with --load-db, after its rows commit -- so a
    manifest's mtime is the second that unit finished. The mtimes are therefore
    a complete, exact completion timeline, kept even when the run's stdout was
    not.
  * Did throughput DECAY or STOP DEAD? A decay (the 10-min histogram sloping
    off, elapsed_s climbing) means a resource degraded under load: the box
    swapping, the DB or Xwave slowing down. A cliff means something wedged at
    once. These have opposite fixes, and the abort message distinguishes
    neither.
  * Was the box the problem? Workers report peak_rss_mb, but a worker only ever
    sees its own. NOTE that under --start-method fork the model is loaded in the
    parent and shared copy-on-write, so these do NOT add up across workers:
    compare the total against `sar -r`, never peak_rss_mb x --workers.

Two things the manifests cannot show you, so do not read them as absent:

  * THE FATAL GAP IS NOT IN THE GAP LIST. Gaps are measured between two
    completions. The stall that triggered the abort ends in the abort, not in a
    manifest, so it leaves no interval to measure -- it shows up only as the
    final silence, printed separately below.
  * THE TAIL IS A SHUTDOWN DRAIN. The abort raises SystemExit from inside the
    `with ProcessPoolExecutor(...)` block, so __exit__ runs shutdown(wait=True)
    and every already-running worker runs to completion, writing its manifest
    AFTER the abort message printed. Completions in the last bucket are
    therefore not progress the parent ever observed.

Reads nothing but the out-dir: no DB, no network, no imports beyond the stdlib,
so it runs on the server against a dead run without reconstructing its
environment.

    python3 offline_stall_forensics.py <out-dir> [--workers 126]
"""
import argparse
import json
import sys
import time
from pathlib import Path

BUCKET_S = 600          # histogram resolution: 10 minutes
EDGE = 20               # units compared head vs tail


def load_manifests(out_dir: Path) -> list:
    """Every manifest, oldest completion first, with its mtime attached."""
    rows = []
    for path in sorted((out_dir / "manifests").glob("unit_*.json")):
        try:
            man = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            # A torn manifest is itself a finding: it is written .tmp then
            # os.replace'd, so an unreadable one means the filesystem, not the
            # runner, lost the write.
            print(f"  UNREADABLE manifest {path.name}: {exc}", file=sys.stderr)
            continue
        man["_mtime"] = path.stat().st_mtime
        rows.append(man)
    rows.sort(key=lambda m: m["_mtime"])
    return rows


def hhmmss(epoch: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(epoch))


def report_gaps(rows: list) -> None:
    """The stall itself: the longest silences between two completions."""
    print("\n=== largest gaps BETWEEN completions ===")
    print("  (the gap that triggered the abort is NOT here -- it ends in the")
    print("   abort, not in a manifest. It is the 'final silence' above.)")
    gaps = sorted(((rows[i + 1]["_mtime"] - rows[i]["_mtime"], i)
                   for i in range(len(rows) - 1)), reverse=True)
    for gap, i in gaps[:8]:
        print(f"  {gap / 60:7.1f} min  after unit {rows[i]['unit']:>7} "
              f"at {hhmmss(rows[i]['_mtime'])}")


def report_throughput(rows: list) -> None:
    """Completions per bucket -- a slope means decay, a cliff means a wedge."""
    t0 = rows[0]["_mtime"]
    buckets: dict = {}
    for man in rows:
        key = int((man["_mtime"] - t0) // BUCKET_S)
        buckets[key] = buckets.get(key, 0) + 1
    print(f"\n=== completions per {BUCKET_S // 60} min "
          f"(sloping off = decay, cliff = wedged) ===")
    for key in sorted(buckets):
        n = buckets[key]
        print(f"  {hhmmss(t0 + key * BUCKET_S)}  {n:>5}  {'#' * min(n, 60)}")


def report_tail(rows: list) -> None:
    """The last completions, with the delay before each.

    Read bottom-up: the run's real death is above the point where these turn
    into the shutdown drain (units that were already running when the parent
    gave up and are merely finishing)."""
    print("\n=== last 15 completions (inter-arrival) ===")
    for i in range(max(1, len(rows) - 15), len(rows)):
        delta = (rows[i]["_mtime"] - rows[i - 1]["_mtime"]) / 60
        print(f"  {hhmmss(rows[i]['_mtime'])}  unit {rows[i]['unit']:>7}  "
              f"+{delta:6.1f} min  elapsed_s={rows[i]['elapsed_s']:>8.1f}")


def report_unit_cost(rows: list, workers: int) -> None:
    """Head vs tail of the run: were units themselves getting more expensive?

    elapsed_s is measured inside the worker, so it excludes queue time -- it
    separates "units got slower" from "units were fine but never got a core"."""
    head, tail = rows[:EDGE], rows[-EDGE:]
    mean = lambda rs, k: sum(r[k] for r in rs) / len(rs)     # noqa: E731
    print(f"\n=== per-unit cost: first {len(head)} vs last {len(tail)} finished ===")
    print(f"  elapsed_s      first {mean(head, 'elapsed_s'):9.1f}s   "
          f"last {mean(tail, 'elapsed_s'):9.1f}s")
    if "peak_rss_mb" not in rows[0]:
        return
    worst = max(rows, key=lambda m: m["peak_rss_mb"])
    print(f"  peak_rss_mb    first {mean(head, 'peak_rss_mb'):9.0f}    "
          f"last {mean(tail, 'peak_rss_mb'):9.0f}")
    print(f"  worst worker   unit {worst['unit']} at {worst['peak_rss_mb']:.0f} MB")
    # Deliberately NOT multiplied by --workers. Under fork the model and every
    # page untouched since the fork are shared, so per-worker RSS double-counts
    # heavily; x126 overstated one real run by 3.7x (662 GB predicted, 177 GB
    # actual). Only the host can answer this, so point at the host.
    print(f"  do NOT multiply by {workers}: under --start-method fork these "
          f"share pages copy-on-write.")
    print("  the real number is `sar -r` (kbmemused) and `free -g` during the run.")


def report_db(rows: list) -> None:
    """With --load-db, a unit commits before its manifest -- so these totals
    only cover units that actually finished."""
    if rows[0].get("db_prob_rows") is None:
        return
    print("\n=== rows committed by finished units ===")
    for key in ("db_prob_rows", "db_feat_rows", "db_xmatch_rows"):
        print(f"  {key:<16} {sum(m.get(key, 0) for m in rows):,}")
    barren = [m["unit"] for m in rows[-50:] if not m.get("db_prob_rows")]
    if barren:
        print(f"  units among the last 50 that committed 0 probability rows: "
              f"{barren}")


def report_errors(rows: list) -> None:
    total = sum(m["n_errors"] for m in rows)
    hit = sum(1 for m in rows if m["n_errors"])
    print(f"\n=== per-oid errors: {total:,} across {hit} unit(s) ===")
    # The manifest samples 20 per unit; errors/unit_*.jsonl has all of them.
    kinds: dict = {}
    for man in rows:
        for err in man.get("errors", []):
            key = (err.get("error") or str(err))[:80]
            kinds[key] = kinds.get(key, 0) + 1
    for key, n in sorted(kinds.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:>6}  {key}")
    if total:
        print("  full list: cat <out-dir>/errors/*.jsonl | jq -r .oid > retry.txt")


def report_inflight(out_dir: Path) -> None:
    """Leftover .tmp files name the units that were mid-write when it died.

    The next run's clean_stale_tmp() removes these, so read them BEFORE
    resuming."""
    tmps = sorted(out_dir.rglob("*.tmp"))
    print(f"\n=== leftover .tmp (units killed mid-write): {len(tmps)} ===")
    for path in tmps[:20]:
        print(f"  {path.relative_to(out_dir)}  "
              f"{hhmmss(path.stat().st_mtime)}")
    if tmps:
        print("  these are deleted by the next run's stale-tmp sweep -- "
              "read them first")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="the run's --out-dir")
    ap.add_argument("--workers", type=int, default=126,
                    help="the run's --workers, for the total-RSS estimate "
                         "(default: %(default)s)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rows = load_manifests(out_dir)
    if not rows:
        print(f"no manifests under {out_dir}/manifests -- wrong --out-dir, or "
              f"the run never finished a single unit", file=sys.stderr)
        return 1

    t0, t_last = rows[0]["_mtime"], rows[-1]["_mtime"]
    stamp = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))  # noqa: E731
    print(f"units finished  : {len(rows):,}")
    print(f"first finished  : {stamp(t0)}")
    print(f"last finished   : {stamp(t_last)}   <- progress stopped here")
    print(f"span            : {(t_last - t0) / 3600:.2f} h")
    print(f"final silence   : {(time.time() - t_last) / 60:.1f} min and counting"
          f"  <- the stall that aborted the run lives HERE, not in the gap list")

    report_gaps(rows)
    report_tail(rows)
    report_throughput(rows)
    report_unit_cost(rows, args.workers)
    report_db(rows)
    report_errors(rows)
    report_inflight(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
