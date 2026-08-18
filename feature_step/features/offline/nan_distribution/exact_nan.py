"""Exact per-feature NaN aggregation for both versions in ONE full scan.
statement_timeout disabled so the seq scan runs to completion (~775GB read).
"""
import json, os, time, pandas as pd, sqlalchemy as sa
from sqlalchemy import text

HERE = os.path.dirname(os.path.abspath(__file__))
p = json.load(open(os.path.join(HERE, "..", "credentials.json")))
e = sa.create_engine(f"postgresql+psycopg2://{p['user']}:{p['password']}@{p['host']}/{p['dbname']}")
OUT = os.path.join(HERE, "nan_per_feature_exact.csv")

t = time.time()
with e.connect() as c:
    c.execute(text("SET statement_timeout = 0"))          # no timeout
    c.execute(text("SET work_mem = '512MB'"))             # help the hash aggregate
    q = text("""
        SELECT version, name, fid, count(*) n,
               sum(case when value is null then 1 else 0 end) n_null
        FROM alerce.feature
        WHERE version IN ('27.5.6','27.5.7a32.dev1')
        GROUP BY version, name, fid
    """)
    df = pd.read_sql_query(q, c)

df['nan_pct'] = 100 * df['n_null'] / df['n']
df.to_csv(OUT, index=False)
print(f"DONE in {(time.time()-t)/60:.1f} min -> {OUT}")
print(df.groupby('version').agg(features=('name','size'), rows=('n','sum'),
                                mean_nan=('nan_pct','mean')).to_string())
