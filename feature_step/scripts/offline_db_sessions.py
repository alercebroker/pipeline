#!/usr/bin/env python
"""What are the run's own database sessions doing, sampled over time?

A unit is an hour of compute and then one burst of upserts, and every worker in
a round starts together, so the writes of a whole round land on the database in
the same minutes. A single sample of pg_stat_activity taken mid-round sees an
idle database and says nothing about the write phase. This samples every
--interval seconds, for --duration seconds (0 = until Ctrl-C), and prints one
line per sample: the run's sessions by state and wait event, how long the
oldest active statement has been running, and what kind of statement it is.
Only sessions of the connecting user are visible without superuser, which is
all the run's sessions and nothing else.

    poetry run python scripts/offline_db_sessions.py \
        --credentials <the run's write credentials> --interval 30 | tee $RUN/db_sessions.log

Read-only: one SELECT on pg_stat_activity per sample.
"""
import argparse
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_run_batch as M  # noqa: E402  (pipeline paths)
from features.offline import db  # noqa: E402

QUERY = text("""
    SELECT state, wait_event_type, wait_event,
           EXTRACT(EPOCH FROM (now() - query_start)) AS age_s,
           left(regexp_replace(query, '\\s+', ' ', 'g'), 40) AS q
    FROM pg_stat_activity
    WHERE usename = current_user AND pid <> pg_backend_pid()
""")


def kind(q: str) -> str:
    q = (q or "").strip().upper()
    for k in ("INSERT", "UPDATE", "SELECT", "COMMIT", "BEGIN", "DELETE"):
        if q.startswith(k):
            return k
    return q[:12] or "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--credentials", default=M.DEFAULT_CREDENTIALS)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
    args = ap.parse_args()

    engine = db._make_engine(args.credentials)
    t_end = time.monotonic() + args.duration if args.duration else None
    print("time(UTC)            sessions  active  idle  idle-in-tx  waits(active)              oldest-active  statements(active)", flush=True)
    while True:
        with engine.connect() as conn:
            rows = conn.execute(QUERY).fetchall()
        state = Counter((r.state or "?") for r in rows)
        active = [r for r in rows if r.state == "active"]
        waits = Counter(f"{r.wait_event_type or 'CPU'}" for r in active)
        kinds = Counter(kind(r.q) for r in active)
        oldest = max((r.age_s or 0) for r in active) if active else 0
        print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}  "
              f"{len(rows):8d}  {state.get('active', 0):6d}  {state.get('idle', 0):4d}  "
              f"{state.get('idle in transaction', 0):10d}  "
              f"{dict(waits) if waits else '-':<26}  {oldest:10.0f}s    {dict(kinds) if kinds else '-'}",
              flush=True)
        if t_end and time.monotonic() >= t_end:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
