#!/usr/bin/env python3
"""
Backfill daemon for index.txt files.

On startup it walks DATA_DIR and generates index.txt for every MJD directory
that doesn't already have one. The scan streams directly to disk so memory
usage is bounded even for directories with tens of millions of files.

After backfill it sleeps for CHECK_INTERVAL seconds and repeats — so any new
MJD directories that appear later (e.g. the first alert of a new night arrives
before the alerts_store writer has had a chance to create its index.txt) are
also handled automatically.

For ongoing writes the alerts_store process itself appends to index.txt on
every batch, so this script is only needed for:
  1. The one-time retroactive backfill of existing data.
  2. Safety net: catch any directory whose index.txt is missing for other reasons.
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
# How often to check for new directories that need backfill (seconds).
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "3600"))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _mjd_dirs() -> list[Path]:
    return sorted(
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "lost+found"
    )


def backfill(mjd_dir: Path) -> None:
    """
    Stream-scan mjd_dir and write index.txt.

    Uses os.scandir() which calls getdents64() in kernel-sized chunks and
    checks is_file() via the d_type field — no extra stat() per file.
    Writes are buffered at 1 MB to amortise the syscall overhead.
    A .tmp file is written first and atomically renamed so a partial scan
    never leaves a corrupt index.txt.
    """
    tmp = mjd_dir / "index.txt.tmp"
    count = 0
    t0 = time.monotonic()

    print(f"[{_now()}] backfill start: {mjd_dir.name}/", flush=True)

    with open(tmp, "w", buffering=1024 * 1024) as out:
        with os.scandir(mjd_dir) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False) and entry.name.endswith(".avro"):
                    out.write(entry.name + "\n")
                    count += 1
                    if count % 500_000 == 0:
                        elapsed = time.monotonic() - t0
                        print(
                            f"  {mjd_dir.name}: {count:,} files … ({elapsed:.0f}s)",
                            flush=True,
                        )

    tmp.rename(mjd_dir / "index.txt")  # atomic on same filesystem
    elapsed = time.monotonic() - t0
    print(
        f"[{_now()}] backfill done:  {mjd_dir.name}/ — "
        f"{count:,} files in {elapsed:.1f}s",
        flush=True,
    )


def write_root_index(dirs: list[Path]) -> None:
    """Write /data/index.txt — one MJD directory name per line."""
    tmp = DATA_DIR / "index.txt.tmp"
    with open(tmp, "w") as f:
        for d in sorted(dirs):
            f.write(d.name + "\n")
    tmp.rename(DATA_DIR / "index.txt")
    print(f"[{_now()}] wrote root index.txt ({len(dirs)} nights)", flush=True)


def run_pass() -> None:
    dirs = _mjd_dirs()
    missing = [d for d in dirs if not (d / "index.txt").exists()]

    if not missing:
        print(f"[{_now()}] all {len(dirs)} dirs indexed, nothing to do", flush=True)
    else:
        print(
            f"[{_now()}] {len(missing)} dir(s) need backfill "
            f"(skipping {len(dirs) - len(missing)} already indexed)",
            flush=True,
        )
        for d in missing:
            # Clean up any leftover .tmp from a previous interrupted run
            tmp = d / "index.txt.tmp"
            if tmp.exists():
                tmp.unlink()
                print(f"  removed stale {tmp.name}", flush=True)
            backfill(d)

    write_root_index(dirs)


if __name__ == "__main__":
    print(f"[{_now()}] indexer starting — DATA_DIR={DATA_DIR}", flush=True)
    while True:
        try:
            run_pass()
        except Exception as exc:
            print(f"[{_now()}] ERROR: {exc}", flush=True)
        print(f"[{_now()}] sleeping {CHECK_INTERVAL}s until next check", flush=True)
        time.sleep(CHECK_INTERVAL)
