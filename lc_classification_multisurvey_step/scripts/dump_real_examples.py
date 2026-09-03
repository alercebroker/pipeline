"""Dump real objects and their production probabilities into a test fixture.

`tests/integration/test_real_data_equivalence.py` replays these through the step
and requires the result to match. Keeping them in a file rather than querying at
test time means the check needs neither the VPN nor database credentials -- only
the model pickle -- so it runs anywhere the model does.

    REAL_DB_CONFIG=$(pwd)/local_config.yaml python scripts/dump_real_examples.py

Writes `tests/integration/data/real_examples.json.gz`. Re-run it when the model
or the feature set changes; the fixture records the object ids it used, so a
refresh is reproducible.

Features are stored as the database spells them -- (feature_name, band, value)
straight from `feature_name_lut` -- not pre-translated to the model's column
names. The translation is the subtle part (hyphens, slashes, the band-12 pair),
so it belongs in the code under test rather than baked into the fixture.
"""
import argparse
import gzip
import json
import os
from datetime import datetime, timezone

import yaml
from sqlalchemy import bindparam, text

from lc_classification_multisurvey_step.db.db import PSQLConnection

ZTF_SID = 0
BHRF_IDS = [5, 6, 7, 8, 9]
CLASSES_PER_OID = 45  # 21 + 3 + 6 + 6 + 9, all five heads
MIN_FEATURES = 190  # of the model's 199; the rest are legitimately absent

OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "tests",
    "integration",
    "data",
    "real_examples.json.gz",
)


def candidate_oids(connection, wanted: int, partitions: int, per_partition: int) -> list:
    """Objects with a nearly complete feature vector and all five heads written.

    Sampled per partition rather than grouped over the whole `feature` table,
    which is partitioned by hash of oid and far too large to aggregate.
    """
    chosen = []
    with connection.session() as session:
        for part in range(partitions):
            if len(chosen) >= wanted:
                break
            sample = [
                int(row[0])
                for row in session.execute(
                    text(
                        f"SELECT DISTINCT oid FROM feature_part_{part} "
                        "WHERE sid = :sid LIMIT :limit"
                    ),
                    {"sid": ZTF_SID, "limit": per_partition},
                )
            ]
            if not sample:
                continue

            rich = [
                int(row[0])
                for row in session.execute(
                    text(
                        "SELECT oid FROM feature WHERE sid = :sid AND oid IN :oids "
                        "GROUP BY oid HAVING count(*) >= :minimum"
                    ).bindparams(bindparam("oids", expanding=True)),
                    {"sid": ZTF_SID, "oids": sample, "minimum": MIN_FEATURES},
                )
            ]
            if not rich:
                continue

            complete = [
                int(row[0])
                for row in session.execute(
                    text(
                        "SELECT oid FROM probability WHERE oid IN :oids "
                        "AND classifier_id IN :classifier_ids "
                        "GROUP BY oid HAVING count(*) = :expected"
                    ).bindparams(
                        bindparam("oids", expanding=True),
                        bindparam("classifier_ids", expanding=True),
                    ),
                    {
                        "oids": rich,
                        "classifier_ids": BHRF_IDS,
                        "expected": CLASSES_PER_OID,
                    },
                )
            ]
            print(
                f"  partition {part}: sampled {len(sample)}, "
                f"{len(rich)} with >={MIN_FEATURES} features, {len(complete)} fully classified"
            )
            chosen.extend(complete)

    return sorted(set(chosen))[:wanted]


def dump(connection, oids: list) -> dict:
    features_statement = text(
        "SELECT f.oid, l.feature_name, f.band, f.value "
        "FROM feature f JOIN feature_name_lut l "
        "  ON l.feature_id = f.feature_id AND l.sid = f.sid "
        "WHERE f.sid = :sid AND f.oid IN :oids"
    ).bindparams(bindparam("oids", expanding=True))

    probability_statement = text(
        "SELECT oid, classifier_id, class_id, probability, ranking, "
        "classifier_version, lastmjd FROM probability "
        "WHERE oid IN :oids AND classifier_id IN :classifier_ids"
    ).bindparams(
        bindparam("oids", expanding=True),
        bindparam("classifier_ids", expanding=True),
    )

    objects: dict = {oid: {"oid": oid, "features": [], "probabilities": []} for oid in oids}
    with connection.session() as session:
        for row in session.execute(
            features_statement, {"sid": ZTF_SID, "oids": oids}
        ).mappings():
            objects[int(row["oid"])]["features"].append(
                [row["feature_name"], int(row["band"]), row["value"]]
            )

        for row in session.execute(
            probability_statement, {"oids": oids, "classifier_ids": BHRF_IDS}
        ).mappings():
            entry = objects[int(row["oid"])]
            entry["probabilities"].append(
                [
                    int(row["classifier_id"]),
                    int(row["class_id"]),
                    float(row["probability"]),
                    int(row["ranking"]),
                ]
            )
            entry["lastmjd"] = float(row["lastmjd"])
            entry["classifier_version"] = int(row["classifier_version"])

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "multisurvey_ztf",
        "sid": ZTF_SID,
        "classifier_ids": BHRF_IDS,
        "objects": [objects[oid] for oid in oids],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--per-partition", type=int, default=400)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()

    with open(os.environ["REAL_DB_CONFIG"]) as handle:
        config = yaml.safe_load(handle)
    connection = PSQLConnection(config["PSQL_CONFIG"], poolclass="NullPool")

    print(f"looking for {args.count} objects...")
    oids = candidate_oids(connection, args.count, args.partitions, args.per_partition)
    print(f"\nselected {len(oids)} objects")

    payload = dump(connection, oids)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    feature_rows = sum(len(o["features"]) for o in payload["objects"])
    probability_rows = sum(len(o["probabilities"]) for o in payload["objects"])
    size = os.path.getsize(args.output)
    print(
        f"wrote {os.path.abspath(args.output)}\n"
        f"  {len(payload['objects'])} objects, {feature_rows} feature rows, "
        f"{probability_rows} probability rows, {size/1024:.0f} KiB gzipped"
    )


if __name__ == "__main__":
    main()
