"""BHRF OutputDTO -> scribe-ready probability rows.

Ported from the offline `features/offline/probability_writer.py` with two
changes: this one is batched (offline raises on a multi-row
frame), and the classifier ids come from the database instead of offline's
pinned `CLASSIFIER_IDS = [5..9]` — only the head *names* are pinned.

"""
import logging

log = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_NAME = "lc_classifier_BHRF_forced_phot"

# Positional against the model's hierarchical output; pinned, not configurable.
HEAD_SUFFIXES = ("", "_top", "_transient", "_stochastic", "_periodic")


def head_names(base_name: str = DEFAULT_CLASSIFIER_NAME) -> list:
    """The five classifier names for `base_name`, in head order."""
    return [f"{base_name}{suffix}" for suffix in HEAD_SUFFIXES]


def classifier_version_to_smallint(version: str) -> int:
    """'2.1.0' -> 210. Strips a '_suffix' on the patch part. 0 if not 3 parts."""
    parts = version.split(".")
    if len(parts) == 3:
        parts[-1] = parts[-1].split("_")[0]
        return int("".join(parts))
    return 0


def iter_head_frames(output_dto, base_name: str = DEFAULT_CLASSIFIER_NAME) -> list:
    """[(classifier_name, frame_or_None)] for the five heads, in head order."""
    hierarchical = getattr(output_dto, "hierarchical", None) or {}
    children = hierarchical.get("children") or {}
    names = head_names(base_name)
    return [
        (names[0], output_dto.probabilities),
        (names[1], hierarchical.get("top")),
        (names[2], children.get("Transient")),
        (names[3], children.get("Stochastic")),
        (names[4], children.get("Periodic")),
    ]


def build_probability_rows(
    output_dto,
    lastmjd_map: dict,
    classifier_ids: dict,
    taxonomy_maps: dict,
    *,
    base_name: str = DEFAULT_CLASSIFIER_NAME,
    version: str,
    sid: int = 0,
) -> list:
    """Batched BHRF OutputDTO -> scribe-ready probability row dicts (all 5 heads).

    Caller contract: each head's frame must have a unique oid index — duplicates
    are not collapsed here and would collide on the probability primary key.
    De-duplication happens upstream, in `input_dto.collapse_by_oid`.

    `classifier_ids` and `taxonomy_maps` come from the DB via
    `db.resolve_classifiers` (design §6.1), which already guarantees every head
    resolves — so they are indexed directly and a miss is a bug, not a condition.

    Per design §8 this raises instead of thinning the output: a class name the
    taxonomy does not know, a NaN probability, or an oid with no lastmjd are
    model/taxonomy/producer faults, and a step that logged and carried on would
    silently lose objects for the life of the deploy.
    """
    if output_dto is None or output_dto.probabilities is None:
        return []

    version_smallint = classifier_version_to_smallint(version)
    rows = []

    for classifier_name, frame in iter_head_frames(output_dto, base_name):
        if frame is None or len(frame) == 0:
            continue

        classifier_id = classifier_ids[classifier_name]
        class_id_of = taxonomy_maps[classifier_id]

        # Class names are the frame's COLUMNS, so an unknown name is a frame-wide
        # model/taxonomy drift, identical for every oid in the batch.
        unknown = sorted(set(frame.columns) - set(class_id_of))
        if unknown:
            raise ValueError(
                f"head '{classifier_name}' (classifier_id={classifier_id}): class names "
                f"{unknown} are absent from the seeded taxonomy; the model and the "
                "taxonomy table disagree"
            )

        nan_rows = frame.isna().any(axis=1)
        if nan_rows.any():
            bad = sorted(int(oid) for oid in frame.index[nan_rows])
            raise ValueError(f"head '{classifier_name}': NaN probabilities for oids {bad}")

        melted = (
            frame.rename_axis("oid")
            .reset_index()
            .melt(id_vars=["oid"], var_name="class_name", value_name="probability")
        )
        melted["oid"] = melted["oid"].astype("int64")

        missing_lastmjd = sorted(set(melted["oid"].tolist()) - set(lastmjd_map))
        if missing_lastmjd:
            raise ValueError(
                f"head '{classifier_name}': no lastmjd for oids {missing_lastmjd}; "
                "probability.lastmjd is NOT NULL"
            )

        melted["ranking"] = (
            melted.groupby("oid")["probability"]
            .rank(ascending=False, method="dense")
            .astype(int)
        )
        melted["lastmjd"] = melted["oid"].map(lastmjd_map)
        melted["class_id"] = melted["class_name"].map(class_id_of)

        rows.extend(
            {
                "oid": int(record["oid"]),
                "sid": int(sid),
                "classifier_id": int(classifier_id),
                "classifier_version": int(version_smallint),
                "class_id": int(record["class_id"]),
                "probability": float(record["probability"]),
                "ranking": int(record["ranking"]),
                "lastmjd": float(record["lastmjd"]),
            }
            for record in melted.to_dict("records")
        )

    return rows
