"""BHRF (Squidward 2.1.0) classifier + taxonomy seed fixture — single source of truth.

Class names/order are md5-verified against the deployed S3 pickle
(``squidward/2.1.0/hierarchical_random_forest_model.pkl``); the transient class is
``SESN`` (not ``SNIbc``). See
``docs/superpowers/specs/2026-07-02-ztf-bhrf-classifier-taxonomy-seed-design.md``.

``ztf_classifier_taxonomy_seed.sql`` is generated from this module via
``render_seed_sql()`` — edit here, never hand-edit the SQL.

Names MUST match the model's ``classes_`` exactly: the multisurvey write path maps
``class_name -> class_id`` by exact string match, so a wrong label yields a garbage
class_id.
"""

CLASSIFIER_VERSION = "2.1.0"
TID = 0  # ZTF

# next-free classifier_id: live max is 4 (four stamp classifiers) -> 5..9.
# RE-VERIFY against live at apply time (another deploy could claim 5+ first).
CLASSIFIER_LUT = [
    {"classifier_id": 5, "classifier_name": "lc_classifier_BHRF_forced_phot",
     "classifier_version": CLASSIFIER_VERSION, "tid": TID},
    {"classifier_id": 6, "classifier_name": "lc_classifier_BHRF_forced_phot_top",
     "classifier_version": CLASSIFIER_VERSION, "tid": TID},
    {"classifier_id": 7, "classifier_name": "lc_classifier_BHRF_forced_phot_transient",
     "classifier_version": CLASSIFIER_VERSION, "tid": TID},
    {"classifier_id": 8, "classifier_name": "lc_classifier_BHRF_forced_phot_stochastic",
     "classifier_version": CLASSIFIER_VERSION, "tid": TID},
    {"classifier_id": 9, "classifier_name": "lc_classifier_BHRF_forced_phot_periodic",
     "classifier_version": CLASSIFIER_VERSION, "tid": TID},
]

# classifier_id -> ordered class names (class_id = order = list position).
# Ordering is the model's classes_ (branches alphabetical; flat = list_of_classes).
TAXONOMY_LUT = {
    5: [  # flat, 21 leaves
        "AGN", "Blazar", "CEP", "CV/Nova", "DSCT", "EA", "EB/EW", "LPV",
        "Microlensing", "Periodic-Other", "QSO", "RRLab", "RRLc", "RSCVn",
        "SESN", "SLSN", "SNII", "SNIIn", "SNIa", "TDE", "YSO",
    ],
    6: ["Periodic", "Stochastic", "Transient"],  # top
    7: ["SESN", "SLSN", "SNII", "SNIIn", "SNIa", "TDE"],  # transient
    8: ["AGN", "Blazar", "CV/Nova", "Microlensing", "QSO", "YSO"],  # stochastic
    9: ["CEP", "DSCT", "EA", "EB/EW", "LPV", "Periodic-Other",
        "RRLab", "RRLc", "RSCVn"],  # periodic
}

SCHEMA = "multisurvey_ztf"


def _sql_str(value: str) -> str:
    """Single-quote a SQL string literal, escaping embedded quotes."""
    return "'" + value.replace("'", "''") + "'"


def render_seed_sql() -> str:
    """Render the idempotent seed SQL from the fixture (single source of truth)."""
    lines = [
        "-- BHRF (Squidward 2.1.0) classifier + taxonomy seed for ZTF (tid=0).",
        "-- Generated from feature_step/features/offline/classifier_taxonomy_lut.py",
        "--   (single source of truth). Do not hand-edit; regenerate from the fixture.",
        "-- Run with:  psql \"<conn>\" -f ztf_classifier_taxonomy_seed.sql",
        "--   (needs INSERT privileges, not readonly_user).",
        "--",
        "-- RE-VERIFY next-free classifier_id against live before applying:",
        "--   SELECT classifier_id, classifier_name, classifier_version, tid",
        f"--     FROM {SCHEMA}.classifier ORDER BY classifier_id;",
        "",
        "-- 1. classifier (5 rows: flat + top + 3 branches)",
        f"INSERT INTO {SCHEMA}.classifier "
        "(classifier_id, classifier_name, classifier_version, tid) VALUES",
    ]
    crows = [
        f"({c['classifier_id']}, {_sql_str(c['classifier_name'])}, "
        f"{_sql_str(c['classifier_version'])}, {c['tid']})"
        for c in CLASSIFIER_LUT
    ]
    lines.append(",\n".join(crows))
    lines.append("ON CONFLICT (classifier_id) DO NOTHING;")
    lines.append("")
    lines.append("-- 2. taxonomy (45 rows: 21 + 3 + 6 + 6 + 9; class_id per-classifier 0-indexed)")
    lines.append(
        f"INSERT INTO {SCHEMA}.taxonomy "
        '(class_id, class_name, "order", classifier_id) VALUES'
    )
    trows = []
    for classifier_id in sorted(TAXONOMY_LUT):
        for class_id, class_name in enumerate(TAXONOMY_LUT[classifier_id]):
            trows.append(
                f"({class_id}, {_sql_str(class_name)}, {class_id}, {classifier_id})"
            )
    lines.append(",\n".join(trows))
    lines.append("ON CONFLICT (class_id, classifier_id) DO NOTHING;")
    lines.append("")
    return "\n".join(lines)
