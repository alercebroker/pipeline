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

        for oid, group in melted.groupby("oid", sort=False):
            oid = int(oid)

            unknown = sorted(set(group["class_name"]) - set(class_id_of))
            if unknown:
                log.error(
                    "oid=%s classifier_id=%s: class names %s absent from the taxonomy; "
                    "dropping this oid's rows for this head",
                    oid,
                    classifier_id,
                    unknown,
                )
                continue

            lastmjd = lastmjd_map.get(oid)
            if lastmjd is None:
                log.error(
                    "oid=%s has no lastmjd; dropping its rows for classifier_id=%s",
                    oid,
                    classifier_id,
                )
                continue

            for record in group.to_dict("records"):
                rows.append(
                    {
                        "oid": oid,
                        "sid": int(sid),
                        "classifier_id": int(classifier_id),
                        "classifier_version": int(version_smallint),
                        "class_id": int(class_id_of[record["class_name"]]),
                        "probability": float(record["probability"]),
                        "ranking": int(record["ranking"]),
                        "lastmjd": float(lastmjd),
                    }
                )

    return rows
