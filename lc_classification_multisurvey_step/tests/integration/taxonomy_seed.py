"""The BHRF 2.1.0 seed the step refuses to start without (design doc §8).

`classifier` and `taxonomy` are the only two tables the step reads, and they are
not in the db-plugins authority file: `_initial_data_pipeline.py` seeds ids 1-4
for the stamp classifiers only, so a freshly created database has nothing for
these five heads. Back-porting them there is tracked separately (design §12);
until that lands, this module is what a local database gets seeded from.

The class names are NOT guesses and must not be edited by hand. They were read
off the 2.1.0 pickle itself -- `list_of_classes` for the flat head and
`model[head].classes_` for the other four -- because the model is the only
authority on what its frames' columns are called. `build_probability_rows`
raises when one of a head's class names is absent from the taxonomy, so a seed
assembled from the class hierarchy hardcoded in
`alerce_classifiers/classifiers/hierarchical_random_forest.py` would crash the
step on its first batch: that hierarchy is stale (it lists SNIbc/SNIIb and RRL, while
2.1.0 emits SESN and RRLab/RRLc).

Regenerate with `scripts/dump_model_taxonomy.py` after any model bump.

Checked against live `multisurvey_ztf` on 2026-09-03: classifier ids 5-9, the
five names, `classifier_version` 2.1.0 and `tid` 0 all match, and every head's
class names, `order` and `class_id` values are identical to the ones below. So
this is not merely a plausible local stand-in — a row built against this seed
carries the same `class_id` it would in production, and the §8 startup
assertions resolve there exactly as they do here.
"""
from lc_classification_multisurvey_step.probabilities import (
    DEFAULT_CLASSIFIER_NAME,
    HEAD_SUFFIXES,
    head_names,
)

CLASSIFIER_VERSION = "2.1.0"

# ZTF. `classifier.tid` is NOT NULL, and is not read by this step.
TID_ZTF = 0

# 1-4 are taken by the stamp classifiers in the db-plugins initial data. 5-9
# matches what the offline reference pinned, but nothing depends on the value:
# the step resolves ids by name at startup (design §6.1).
FIRST_CLASSIFIER_ID = 5

# suffix -> class names, in the model's own order, which becomes taxonomy."order".
HEAD_CLASSES = {
    "": [
        "AGN",
        "Blazar",
        "CEP",
        "CV/Nova",
        "DSCT",
        "EA",
        "EB/EW",
        "LPV",
        "Microlensing",
        "Periodic-Other",
        "QSO",
        "RRLab",
        "RRLc",
        "RSCVn",
        "SESN",
        "SLSN",
        "SNII",
        "SNIIn",
        "SNIa",
        "TDE",
        "YSO",
    ],
    "_top": ["Periodic", "Stochastic", "Transient"],
    "_transient": ["SESN", "SLSN", "SNII", "SNIIn", "SNIa", "TDE"],
    "_stochastic": ["AGN", "Blazar", "CV/Nova", "Microlensing", "QSO", "YSO"],
    "_periodic": [
        "CEP",
        "DSCT",
        "EA",
        "EB/EW",
        "LPV",
        "Periodic-Other",
        "RRLab",
        "RRLc",
        "RSCVn",
    ],
}

# Rows one batch of N distinct oids is expected to yield: every class of every
# head, for every oid.
CLASSES_PER_OID = sum(len(classes) for classes in HEAD_CLASSES.values())


def classifier_ids(base_name: str = DEFAULT_CLASSIFIER_NAME) -> dict:
    """{classifier_name: classifier_id}, in head order."""
    return {
        name: FIRST_CLASSIFIER_ID + index
        for index, name in enumerate(head_names(base_name))
    }


def classifier_rows(base_name: str = DEFAULT_CLASSIFIER_NAME) -> list:
    return [
        {
            "classifier_id": classifier_id,
            "classifier_name": name,
            "classifier_version": CLASSIFIER_VERSION,
            "tid": TID_ZTF,
        }
        for name, classifier_id in classifier_ids(base_name).items()
    ]


def taxonomy_maps(base_name: str = DEFAULT_CLASSIFIER_NAME) -> dict:
    """{classifier_id: {class_name: class_id}}, as `db.resolve_classifiers` returns it.

    Lets a test build probability rows without a database. Sound only because
    this seed was checked against live and matches it exactly (see the module
    docstring) -- otherwise the ids here would be fiction.
    """
    maps: dict = {}
    for row in taxonomy_rows(base_name):
        maps.setdefault(row["classifier_id"], {})[row["class_name"]] = row["class_id"]
    return maps


def taxonomy_rows(base_name: str = DEFAULT_CLASSIFIER_NAME) -> list:
    ids = classifier_ids(base_name)
    rows = []
    for suffix, name in zip(HEAD_SUFFIXES, head_names(base_name)):
        for order, class_name in enumerate(HEAD_CLASSES[suffix]):
            rows.append(
                {
                    "class_id": order,
                    "class_name": class_name,
                    "order": order,
                    "classifier_id": ids[name],
                }
            )
    return rows
