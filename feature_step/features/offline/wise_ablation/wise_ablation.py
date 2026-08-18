"""WISE-NaN ablation on BHRF 2.1.0.

For a random sample of objects that DO have WISE (stored 27.5.6 vector, W1-W2
populated), predict the flat + top-head class twice from the SAME vector:
  baseline : WISE colors as stored (present)
  ablated  : the 11 WISE color features forced to NaN (dev1's no-xmatch condition)

Everything else identical -> the class change is purely the WISE-NaN effect.
"""
import os, sys, time
from pathlib import Path
PIPE = Path("/home/fandrades/desktop/pipeline")
for p in (PIPE/"feature_step", PIPE/"lc_classifier", PIPE/"libs"/"idmapper",
          PIPE/"libs"/"apf", PIPE/"libs"/"xmatch_client", PIPE/"alerce_classifiers"):
    sys.path.insert(0, str(p))
os.environ.setdefault("MODEL_PATH",
    "/home/fandrades/desktop/alerce_models/squidward/2.1.0/hierarchical_random_forest_model.pkl")

import numpy as np, pandas as pd
from sqlalchemy import text
from features.offline import db
from features.offline.model_feature_list import MODEL_FEATURE_LIST
from features.offline.classify import load_squidward_model, features_message_to_dto, _empty_output
from features.offline.probability_compare import offline_dto_to_series

CRED = str(PIPE/"feature_step"/"features"/"offline"/"credentials.json")
SCRATCH = Path(__file__).parent
FLAT = "lc_classifier_BHRF_forced_phot"
TOP  = "lc_classifier_BHRF_forced_phot_top"
V6 = "27.5.6"
WISE = ["W1_W2", "W2_W3", "W3_W4", "g_W1", "g_W2", "g_W3", "g_W4", "r_W1", "r_W2", "r_W3", "r_W4"]
FID = {1: "_1", 2: "_2", 12: "_12", 0: ""}
MODEL_SET = list(MODEL_FEATURE_LIST)
N_TARGET = 5000

def wide(name, fid):
    return (name.replace("Power_rate_1/", "Power_rate_1_") + FID.get(int(fid), "")).replace("-", "_")

def predict(oid, wdict, model):
    dto = features_message_to_dto({"oid": oid, "features": wdict})
    can, _ = model.can_predict(dto)
    out = model.predict(dto) if can else _empty_output()
    ser = offline_dto_to_series(out)
    flat = str(ser[FLAT].astype(float).idxmax()) if FLAT in ser and len(ser[FLAT]) else None
    top  = str(ser[TOP].astype(float).idxmax())  if TOP  in ser and len(ser[TOP])  else None
    return flat, top

model, mname, mver = load_squidward_model()
assert "SESN" in model.model.list_of_classes and "SNIbc" not in model.model.list_of_classes, "WRONG MODEL"
print(f"model {mname} v{mver} OK", flush=True)

eng = db._make_engine(CRED)
t0 = time.time()
with eng.connect() as c:
    c.execute(text("SET statement_timeout='600s'"))
    cand = pd.read_sql_query(text(
        "SELECT DISTINCT oid FROM alerce.feature TABLESAMPLE SYSTEM (0.02) WHERE version = :v"),
        c, params={"v": V6})["oid"].tolist()
print(f"[{time.time()-t0:.0f}s] sampled {len(cand)} oids with {V6}", flush=True)
if len(cand) > N_TARGET:
    step = len(cand)/N_TARGET
    cand = [cand[int(i*step)] for i in range(N_TARGET)]

with eng.connect() as c:
    c.execute(text("SET statement_timeout='600s'"))
    feats = pd.read_sql_query(text(
        "SELECT oid,name,fid,value FROM alerce.feature WHERE oid = ANY(:o) AND version = :v"),
        c, params={"o": cand, "v": V6})
print(f"fetched {len(feats)} feature rows for {feats.oid.nunique()} oids", flush=True)

rows = []
for oid, g in feats.groupby("oid"):
    base = {wide(nm, fid): (np.nan if v is None else float(v))
            for nm, fid, v in zip(g["name"], g["fid"], g["value"])}
    for m in MODEL_SET:
        base.setdefault(m, np.nan)
    if not np.isfinite(base.get("W1_W2", np.nan)):   # keep only WISE-populated objects
        continue
    abl = dict(base)
    for w in WISE:
        abl[w] = np.nan
    try:
        fb, tb = predict(oid, base, model)
        fa, ta = predict(oid, abl, model)
        rows.append({"oid": oid, "flat_base": fb, "flat_abl": fa, "top_base": tb, "top_abl": ta})
    except Exception as e:
        rows.append({"oid": oid, "err": f"{type(e).__name__}:{str(e)[:40]}"})
    if len(rows) % 250 == 0:
        pd.DataFrame(rows).to_csv(SCRATCH/"wise_ablation.csv", index=False)
        print(f"  [{len(rows)}]", flush=True)

df = pd.DataFrame(rows)
df.to_csv(SCRATCH/"wise_ablation.csv", index=False)
ok = df[df["flat_base"].notna() & df["flat_abl"].notna()]
print(f"\nWISE-populated objects scored: {len(ok)}", flush=True)
print(f"FLAG unchanged: flat {100*(ok.flat_base==ok.flat_abl).mean():.1f}%  |  top {100*(ok.top_base==ok.top_abl).mean():.1f}%")
print("\n-- TOP head marginal --")
print(pd.DataFrame({"base": ok.top_base.value_counts(), "ablated(WISE=NaN)": ok.top_abl.value_counts()}).fillna(0).astype(int).to_string())
print("\n-- TOP head transitions (base -> ablated), changed only --")
chg = ok[ok.top_base != ok.top_abl]
print(chg.groupby(["top_base", "top_abl"]).size().sort_values(ascending=False).to_string())
print("\n-- FLAT transitions (base -> ablated), top 15 changed --")
cf = ok[ok.flat_base != ok.flat_abl]
print(cf.groupby(["flat_base", "flat_abl"]).size().sort_values(ascending=False).head(15).to_string())
print("DONE_MARKER", flush=True)
