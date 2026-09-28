#!/usr/bin/env python
"""Why did --eligible drop names? Sample the list and look them up by primary key.

offline_reprocessed_oids.py --eligible reports one number for "absent from
<schema>.object or under the n_det cut". The two mean different things: absent
means the reprocessed alerts for that object have not been ingested; under the
cut means a single-detection object the run would not classify anyway. This
takes a random sample of the names, looks them up on object's primary key
(cheap at any table size) and prints the split.

    poetry run python scripts/offline_reprocessed_probe.py \
        $RUN/oids/ztf_reprocesados_oids.txt --sample 2000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_reprocessed_oids as M  # noqa: E402
from features.offline import db  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ztf_list")
    ap.add_argument("--column", default="oid")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--credentials", default=M.DEFAULT_CREDENTIALS)
    ap.add_argument("--min-n-det", type=int, default=M.DEFAULT_MIN_N_DET)
    args = ap.parse_args()

    names = np.asarray(M.read_ztf_ids(args.ztf_list, args.column))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(names), min(args.sample, len(names)), replace=False)
    sample = M.build_oid_array(names[pick])
    n = len(sample)

    engine = db._make_engine(args.credentials)
    with engine.connect() as conn:
        got = pd.read_sql(
            text(f"SELECT oid, n_det, lastmjd, updated_date FROM {db.SCHEMA}.object "
                 "WHERE sid = :sid AND oid = ANY(:oids)"),
            conn, params={"oids": [int(x) for x in sample], "sid": db.SID})

    absent = n - len(got)
    under = int((got.n_det < args.min_n_det).sum())
    ok = int((got.n_det >= args.min_n_det).sum())
    print(f"sampled {n:,} names from {args.ztf_list}")
    print(f"  absent from {db.SCHEMA}.object : {absent:,} ({absent / n:.1%})")
    print(f"  present, n_det < {args.min_n_det}          : {under:,} ({under / n:.1%})")
    print(f"  present, n_det >= {args.min_n_det} (eligible): {ok:,} ({ok / n:.1%})")
    if len(got):
        print("\nlastmjd of the present objects:")
        print(got.lastmjd.describe().to_string())
        print("\nupdated_date of the present objects:")
        print(got.updated_date.describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
