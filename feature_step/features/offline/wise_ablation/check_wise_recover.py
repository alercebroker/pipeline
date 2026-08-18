"""Do WISE-null-in-DB objects have a recoverable AllWISE match via our client?

Two sets, both currently WISE-null in the live DB (multisurvey_ztf.xmatch empty):
  A. WISE-PRESENT in 27.5.6  -> what dev1/no-xmatch dropped; should be recoverable.
  B. WISE-NULL   in 27.5.6  -> the ~20% legacy nulls; are they real, or recoverable?
Cone-search Xwave at several radii and report the recovery fraction per set.
"""
import sys, time
from pathlib import Path
PIPE = Path("/home/fandrades/desktop/pipeline")
for p in (PIPE/"feature_step", PIPE/"libs"/"xmatch_client"):
    sys.path.insert(0, str(p))
import numpy as np, pandas as pd
from sqlalchemy import text
from features.offline import db, xmatch

CRED = str(PIPE/"feature_step"/"features"/"offline"/"credentials.json")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
RADII = [1.005, 1.5, 3.0]
eng = db._make_engine(CRED)

def sample(null_clause, n):
    with eng.connect() as c:
        c.execute(text("SET statement_timeout='300s'"))
        oids = pd.read_sql_query(text(
            f"SELECT oid, value FROM alerce.feature TABLESAMPLE SYSTEM (0.1) "
            f"WHERE version='27.5.6' AND name='W1-W2' AND {null_clause} LIMIT :n"),
            c, params={"n": n})
        co = pd.read_sql_query(text(
            "SELECT oid, meanra, meandec FROM alerce.object WHERE oid = ANY(:o)"),
            c, params={"o": oids["oid"].tolist()})
    df = oids.merge(co, on="oid").reset_index(drop=True)
    df["key"] = df.index                       # integer label for the client
    return df

def recover(df, radius):
    aw = xmatch.compute_allwise(df["key"].tolist(), df["meanra"].tolist(),
                                df["meandec"].tolist(),
                                base_url=xmatch.DEFAULT_XMATCH_URL, radius=radius)
    if not len(aw):
        return 0
    return int(aw["W1"].notna().sum())

for label, clause in [("A WISE-PRESENT in 27.5.6", "value IS NOT NULL"),
                      ("B WISE-NULL    in 27.5.6", "value IS NULL")]:
    df = sample(clause, N)
    print(f"\n=== set {label}  (n={len(df)}) ===", flush=True)
    for r in RADII:
        t0 = time.time()
        got = recover(df, r)
        print(f"  radius {r:>5}\":  {got:3d}/{len(df)} recovered a real WISE mag "
              f"({100*got/len(df):.0f}%)  [{time.time()-t0:.0f}s]", flush=True)
print("\nDONE")
