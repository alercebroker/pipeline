"""% de valores faltantes por feature, abierto por n_det.

Mismo conteo que missing_per_feature.py (faltante = fila ausente, ver README),
pero con los objetos particionados en bins de `multisurvey_ztf.object.n_det`.

Dos ejes:
  --axis object  (default) n_det total del objeto, todas las bandas juntas.
  --axis band    n_det de la banda de la feature (multisurvey_ztf.magstat),
                 solo para features de band 1/2 — es la compuerta que de verdad
                 ven turbo-FATS y MHPS. Un objeto sin fila en magstat para esa
                 banda cuenta como n_det = 0 en ese eje.

Uso:
    <venv>/bin/python missing_by_ndet.py [--axis object|band] [--csv PATH]
"""
import argparse
import csv
import json
import os

import sqlalchemy as sa
from sqlalchemy import text

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CREDENTIALS = os.path.join(HERE, os.pardir, "credentials.json")

SCHEMA = os.getenv("OFFLINE_DB_SCHEMA", "multisurvey_ztf")
SID = 0
BAND_LABEL = {0: "-", 1: "g", 2: "r", 12: "gr"}

# (etiqueta, limite superior inclusivo o None)
BINS = [("2-4", 4), ("5-9", 9), ("10-19", 19), ("20-49", 49), ("50-99", 99), ("100+", None)]
BAND_BINS = [("0", 0), ("1-4", 4), ("5-9", 9), ("10-19", 19), ("20-49", 49), ("50+", None)]


def _case(col, bins):
    parts = []
    for label, hi in bins:
        if hi is not None:
            parts.append(f"when {col} <= {hi} then '{label}'")
    return "case " + " ".join(parts) + f" else '{bins[-1][0]}' end"


def _engine(credentials):
    with open(credentials, "r", encoding="utf-8") as fh:
        p = json.load(fh)
    return sa.create_engine(
        f"postgresql+psycopg2://{p['user']}:{p['password']}"
        f"@{p['host']}:{p.get('port', 5432)}/{p['dbname']}"
    )


def by_object(c, lut):
    """Eje: object.n_det total."""
    bin_case = _case("o.n_det", BINS)
    denom = {
        r[0]: r[1]
        for r in c.execute(
            text(
                f"with objs as (select distinct oid from {SCHEMA}.feature "
                f"            where sid = {SID} and feature_id = 124) "
                f"select {bin_case} b, count(*) "
                f"from objs join {SCHEMA}.object o on o.oid = objs.oid and o.sid = {SID} "
                f"group by 1"
            )
        )
    }
    rows = list(
        c.execute(
            text(
                f"with objs as (select o.oid, {bin_case} b from {SCHEMA}.object o "
                f"              where o.sid = {SID} and o.oid in "
                f"                    (select distinct oid from {SCHEMA}.feature "
                f"                     where sid = {SID} and feature_id = 124)) "
                f"select f.feature_id, f.band, objs.b, count(distinct f.oid) "
                f"from {SCHEMA}.feature f join objs on objs.oid = f.oid "
                f"where f.sid = {SID} group by 1, 2, 3"
            )
        )
    )
    return [lb for lb, _ in BINS], denom, rows


def by_band(c, lut):
    """Eje: magstat.n_det de la banda de la feature (solo band 1 y 2)."""
    bin_case = _case("coalesce(m.n_det, 0)", BAND_BINS)
    # Denominador por (banda, bin): TODOS los objetos del censo, con o sin
    # magstat en esa banda (los que no tienen caen en el bin '0').
    denom = {
        (r[0], r[1]): r[2]
        for r in c.execute(
            text(
                f"with objs as (select distinct oid from {SCHEMA}.feature "
                f"              where sid = {SID} and feature_id = 124), "
                f"     grid as (select objs.oid, b.band from objs cross join "
                f"              (select 1 band union all select 2) b) "
                f"select grid.band, {bin_case} bin, count(*) "
                f"from grid left join {SCHEMA}.magstat m "
                f"       on m.oid = grid.oid and m.sid = {SID} and m.band = grid.band "
                f"group by 1, 2"
            )
        )
    }
    rows = [
        (r[0], r[1], r[2], r[3])
        for r in c.execute(
            text(
                f"with objs as (select distinct oid from {SCHEMA}.feature "
                f"              where sid = {SID} and feature_id = 124), "
                f"     ms as (select objs.oid, b.band, "
                f"                   (select {bin_case} from {SCHEMA}.magstat m "
                f"                    where m.oid = objs.oid and m.sid = {SID} "
                f"                      and m.band = b.band) bin "
                f"            from objs cross join (select 1 band union all select 2) b) "
                f"select f.feature_id, f.band, coalesce(ms.bin, '0'), count(distinct f.oid) "
                f"from {SCHEMA}.feature f join ms on ms.oid = f.oid and ms.band = f.band "
                f"where f.sid = {SID} and f.band in (1, 2) group by 1, 2, 3"
            )
        )
    ]
    return [lb for lb, _ in BAND_BINS], denom, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=("object", "band"), default="object")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--csv")
    args = ap.parse_args()
    csv_path = args.csv or os.path.join(HERE, f"missing_by_ndet_{args.axis}.csv")

    engine = _engine(args.credentials)
    with engine.connect() as c:
        lut = {
            r[0]: r[1]
            for r in c.execute(
                text(f"select feature_id, feature_name from {SCHEMA}.feature_name_lut where sid = {SID}")
            )
        }
        labels, denom, counts = (by_object if args.axis == "object" else by_band)(c, lut)

    # {(feature_id, band): {bin: n}}
    present = {}
    for fid, band, b, n in counts:
        present.setdefault((fid, band), {})[b] = n

    def total_for(band, b):
        return denom[b] if args.axis == "object" else denom[(band, b)]

    out = []
    for (fid, band), per_bin in present.items():
        row = [lut.get(fid, f"id{fid}"), fid, band, BAND_LABEL.get(band, band)]
        for b in labels:
            t = total_for(band, b)
            row.append(round(100.0 * (t - per_bin.get(b, 0)) / t, 2) if t else None)
        out.append(row)
    out.sort(key=lambda r: (r[0], r[2]))

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "feature_id", "band", "band_label"] + [f"pct_missing_{b}" for b in labels])
        w.writerows(out)

    print(f"eje: {args.axis}")
    if args.axis == "object":
        print("objetos por bin: " + ", ".join(f"{b}={denom[b]}" for b in labels))
    else:
        for band in (1, 2):
            print(f"objetos por bin (band {BAND_LABEL[band]}): "
                  + ", ".join(f"{b}={denom[(band, b)]}" for b in labels))
    head = f"{'feature':<30}{'band':>5}" + "".join(f"{b:>9}" for b in labels)
    print("\n" + head)
    for row in out:
        print(f"{row[0]:<30}{row[3]:>5}" + "".join(
            f"{v:>9.2f}" if v is not None else f"{'-':>9}" for v in row[4:]))
    print(f"\nCSV -> {csv_path}")


if __name__ == "__main__":
    main()
