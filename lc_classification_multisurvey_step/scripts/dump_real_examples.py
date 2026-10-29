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

`updated_date` is carried per row, because it is not decoration. The upsert is
ON CONFLICT (oid, sid, feature_id, band) DO UPDATE ... updated_date = now(), so
it only touches rows the newest computation produced: a feature computed in an
earlier pass but not in the latest one keeps its old row, old value and old
date, while every other row for that object moves on. Such a row is a leftover
from a superseded computation -- the classifier saw NaN for it -- so the test
keeps only each object's most recent date. Recorded rather than filtered here so
the evidence stays in the fixture.
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

# The only condition, and it is not a quality filter: an object with no BHRF
# probability rows has nothing to compare against. Everything else is taken as
# it comes -- notably objects with a sparse feature vector, which are the ones
# that exercise the NaN handling, and which an earlier version of this script
# wrongly excluded with a minimum-feature-count filter.

OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "tests",
    "integration",
    "data",
    "real_examples.json.gz",
)


def candidate_oids(connection, wanted: int, partitions: int, per_partition: int) -> list:
    """Objects that have ZTF features and BHRF probabilities, taken as they come.

    Sampled per partition rather than grouped over the whole `feature` table,
    which is partitioned by hash of oid and far too large to aggregate. The walk
    spreads over every partition and takes an even share from each, so the set
    is not drawn from one corner of the table.
    """
    chosen: list = []
    share = max(1, wanted // max(1, partitions))

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

            classified = [
                int(row[0])
                for row in session.execute(
                    text(
                        "SELECT DISTINCT oid FROM probability WHERE oid IN :oids "
                        "AND classifier_id IN :classifier_ids"
                    ).bindparams(
                        bindparam("oids", expanding=True),
                        bindparam("classifier_ids", expanding=True),
                    ),
                    {"oids": sample, "classifier_ids": BHRF_IDS},
                )
            ]
            take = classified[: share if len(chosen) + share < wanted else wanted - len(chosen)]
            print(
                f"  partition {part}: sampled {len(sample)}, "
                f"{len(classified)} with BHRF rows, taking {len(take)}"
            )
            chosen.extend(take)

    return sorted(set(chosen))[:wanted]


def dump(connection, oids: list) -> dict:
    features_statement = text(
        "SELECT f.oid, l.feature_name, f.band, f.value, f.updated_date "
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
                [row["feature_name"], int(row["band"]), row["value"],
                 str(row["updated_date"])]
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
