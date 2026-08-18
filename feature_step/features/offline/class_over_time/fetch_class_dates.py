"""Fetch a random sample of BHRF-classified objects with their stored class
(top 3-way head + flat leaf) and their light-curve dates (firstmjd/lastmjd),
for looking at class distribution across dates. Pure DB, no model."""
import json, os, time, pandas as pd, sqlalchemy as sa
from sqlalchemy import text

HERE = os.path.dirname(os.path.abspath(__file__))
p = json.load(open(os.path.join(HERE, "..", "credentials.json")))
e = sa.create_engine(f"postgresql+psycopg2://{p['user']}:{p['password']}@{p['host']}/{p['dbname']}")
SCRATCH = HERE
TOP  = "alerce.lc_classifier_bhrf_forced_phot_top"
FLAT = "alerce.lc_classifier_bhrf_forced_phot"

def sample_class(tbl, frac):
    with e.connect() as c:
        c.execute(text("SET statement_timeout='300s'"))
        return pd.read_sql_query(text(
            f"SELECT oid, class_name FROM {tbl} TABLESAMPLE SYSTEM ({frac}) WHERE ranking = 1"), c)

def add_dates(df):
    oids = df["oid"].tolist()
    out = []
    with e.connect() as c:
        c.execute(text("SET statement_timeout='300s'"))
        for i in range(0, len(oids), 20000):
            chunk = oids[i:i+20000]
            out.append(pd.read_sql_query(text(
                "SELECT oid, firstmjd, lastmjd FROM alerce.object WHERE oid = ANY(:o)"),
                c, params={"o": chunk}))
    obj = pd.concat(out, ignore_index=True)
    m = df.merge(obj, on="oid", how="inner")
    # MJD -> calendar date (MJD 40587 == 1970-01-01)
    m["first_date"] = pd.to_datetime(m["firstmjd"] - 40587, unit="D")
    m["last_date"]  = pd.to_datetime(m["lastmjd"]  - 40587, unit="D")
    return m

t0 = time.time()
top = add_dates(sample_class(TOP, 3.0))
top.to_csv(f"{SCRATCH}/class_dates_top.csv", index=False)
print(f"[{time.time()-t0:.0f}s] TOP  n={len(top)}  classes={sorted(top.class_name.unique())}")
print(top.class_name.value_counts().to_string())

flat = add_dates(sample_class(FLAT, 1.0))
flat.to_csv(f"{SCRATCH}/class_dates_flat.csv", index=False)
print(f"\n[{time.time()-t0:.0f}s] FLAT n={len(flat)}  n_classes={flat.class_name.nunique()}")
print(flat.class_name.value_counts().to_string())
print(f"\nfirst_date range: {top.first_date.min().date()} .. {top.first_date.max().date()}")
print(f"last_date  range: {top.last_date.min().date()} .. {top.last_date.max().date()}")
print("DONE_MARKER")
