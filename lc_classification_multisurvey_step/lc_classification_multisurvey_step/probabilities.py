"""BHRF OutputDTO -> scribe-ready probability rows.

Ported from the offline reference `features/offline/probability_writer.py`, with
two deliberate changes (design doc §5):

  - the offline builder is strictly per-oid and raises on a multi-row frame; this
    one is batched, so it melts by oid;
  - offline pins `CLASSIFIER_IDS = [5..9]`; here the ids come from the database
    and only the head *names* are pinned (design doc §6).

Pure: no database, no alerce_classifiers, no apf. `output_dto` is duck-typed —
anything with `.probabilities` and `.hierarchical` works.
"""
import logging

log = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_NAME = "lc_classifier_BHRF_forced_phot"
DEFAULT_CLASSIFIER_VERSION = "2.1.0"

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
    version: str = DEFAULT_CLASSIFIER_VERSION,
    sid: int = 0,
) -> list:
    """Batched BHRF OutputDTO -> scribe-ready probability row dicts (all 5 heads).

    Parameters
    ----------
    output_dto : anything with `.probabilities` and `.hierarchical`, or None.
    Caller contract: each head's frame must have a unique oid index. Duplicate
    oids are not collapsed here and would emit rows colliding on the probability
    primary key; de-duplication happens upstream when the features frame is built.
    lastmjd_map : {oid: lastmjd}. An oid missing from it is dropped and logged —
        `probability.lastmjd` is NOT NULL.
    classifier_ids : {classifier_name: classifier_id}, from the DB (design §6.1).
    taxonomy_maps : {classifier_id: {class_name: class_id}}, from the DB.

    Per design §8, problems detectable only per-batch are logged and drop the
    affected (oid, head) rows rather than killing the batch. Startup problems are
    the caller's job (`db.resolve_classifiers`).
    """
    if output_dto is None or output_dto.probabilities is None:
        return []

    version_smallint = classifier_version_to_smallint(version)
    rows = []

    for classifier_name, frame in iter_head_frames(output_dto, base_name):
        if frame is None or len(frame) == 0:
            continue

        classifier_id = classifier_ids.get(classifier_name)
        if classifier_id is None:
            log.error(
                "no classifier_id resolved for head '%s'; dropping %d rows for this head",
                classifier_name,
                len(frame),
            )
            continue

        class_id_of = taxonomy_maps.get(classifier_id)
        if not class_id_of:
            log.error(
                "no taxonomy map for classifier_id=%s (head '%s'); dropping this head",
                classifier_id,
                classifier_name,
            )
            continue

        # Class names are the frame's COLUMNS, so an unknown class name is a
        # frame-wide model/taxonomy drift, never a per-oid condition — check once
        # per head rather than once per oid.
        unknown = sorted(set(frame.columns) - set(class_id_of))
        if unknown:
            log.error(
                "classifier_id=%s (head '%s'): class names %s absent from the taxonomy; "
                "dropping this head for all %d oids in the batch",
                classifier_id,
                classifier_name,
                unknown,
                len(frame),
            )
            continue

        melted = (
            frame.rename_axis("oid")
            .reset_index()
            .melt(id_vars=["oid"], var_name="class_name", value_name="probability")
        )
        melted["ranking"] = (
            melted.groupby("oid")["probability"]
            .rank(ascending=False, method="dense")
            .astype(int)
        )
        melted["oid"] = melted["oid"].astype("int64")
        melted["lastmjd"] = melted["oid"].map(lastmjd_map)

        missing_lastmjd = melted["lastmjd"].isna()
        if missing_lastmjd.any():
            dropped = sorted(set(melted.loc[missing_lastmjd, "oid"]))
            log.error(
                "oids %s have no lastmjd; dropping their rows for classifier_id=%s",
                dropped,
                classifier_id,
            )
            melted = melted[~missing_lastmjd]
            if melted.empty:
                continue

        melted["class_id"] = melted["class_name"].map(class_id_of)

        for record in melted.to_dict("records"):
            rows.append(
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
            )

    return rows
