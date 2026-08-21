"""% de valores faltantes por feature en multisurvey_ztf.feature.

A diferencia de `alerce.feature` (donde el NaN se guarda como NULL, ver
../nan_distribution/README.md), aca los NaN/inf NUNCA se insertan:
`features.utils.parsers.prepare_ao_features_for_db` filtra `value.notna()`.
Verificado: 0 filas con `value IS NULL` sobre 3,442,717.

=> "missing" == la fila (oid, feature_id, band) no existe.

Denominador: los oids que tienen Coordinate_x/y/z (feature_id 124/125/126,
band 0). Se emiten para todo objeto que llega a escribirse, sin depender de la
fotometria, asi que son el censo de objetos del run.

Uso:
    <venv>/bin/python missing_per_feature.py [--credentials PATH] [--csv PATH]
"""
import argparse
import csv
import json
import os

import sqlalchemy as sa
from sqlalchemy import text

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CREDENTIALS = os.path.join(HERE, os.pardir, "credentials.json")
DEFAULT_CSV = os.path.join(HERE, "missing_per_feature.csv")

SCHEMA = os.getenv("OFFLINE_DB_SCHEMA", "multisurvey_ztf")
SID = 0  # ZTF
COORDINATE_IDS = (124, 125, 126)  # Coordinate_x / _y / _z
BAND_LABEL = {0: "-", 1: "g", 2: "r", 12: "gr"}


def fetch(credentials: str):
    with open(credentials, "r", encoding="utf-8") as fh:
        p = json.load(fh)
    engine = sa.create_engine(
        f"postgresql+psycopg2://{p['user']}:{p['password']}"
        f"@{p['host']}:{p.get('port', 5432)}/{p['dbname']}"
    )
    with engine.connect() as c:
        # feature_name_lut trae ZTF (sid=0) y LSST (sid=1) con feature_id que se
        # solapan; sin el filtro por sid los nombres salen cruzados.
        lut = {
            r[0]: r[1]
            for r in c.execute(
                text(f"select feature_id, feature_name from {SCHEMA}.feature_name_lut where sid = :sid"),
                {"sid": SID},
            )
        }
        total = c.execute(
            text(
                f"select count(distinct oid) from {SCHEMA}.feature "
                f"where sid = :sid and feature_id in :coords"
            ),
            {"sid": SID, "coords": COORDINATE_IDS},
        ).scalar()
        counts = list(
            c.execute(
                text(
                    f"select feature_id, band, count(distinct oid) "
                    f"from {SCHEMA}.feature where sid = :sid group by 1, 2"
                ),
                {"sid": SID},
            )
        )
    return lut, total, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args()

    lut, total, counts = fetch(args.credentials)
    rows = sorted(
        (
            (lut.get(fid, f"id{fid}"), fid, band, n, 100.0 * (total - n) / total)
            for fid, band, n in counts
        ),
        key=lambda r: (-r[4], r[0], r[2]),
    )

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "feature_id", "band", "band_label", "n_objects", "pct_missing"])
        for name, fid, band, n, pct in rows:
            w.writerow([name, fid, band, BAND_LABEL.get(band, band), n, f"{pct:.2f}"])

    print(f"objetos (denominador, Coordinate_x/y/z): {total}")
    print(f"pares (feature, band): {len(rows)}\n")
    print(f"{'feature':<30}{'band':>5}{'n_obj':>9}{'missing%':>10}")
    for name, _fid, band, n, pct in rows:
        print(f"{name:<30}{BAND_LABEL.get(band, band):>5}{n:>9}{pct:>10.2f}")
    print(f"\nmedia missing%: {sum(r[4] for r in rows) / len(rows):.2f}")
    print(f"CSV -> {args.csv}")


if __name__ == "__main__":
    main()
