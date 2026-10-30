"""Model output -> messages for the daily topic.

Same format as the legacy lc_classification_step (output_ztf.avsc):
features, probabilities, winning class and hierarchical probabilities.
Only difference: `oid` is the multisurvey id (as a string).
"""
import math


def _clean(value):
    """NaN -> None and numpy -> python, like the legacy step."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _row(frame, oid) -> dict:
    return {str(k): float(v) for k, v in frame.loc[oid].items()}


def build_output_messages(output_dto, collapsed: dict) -> list:
    """One message per classified oid; features come from `collapsed`."""
    if output_dto is None or output_dto.probabilities is None:
        return []
    probabilities = output_dto.probabilities
    if len(probabilities) == 0:
        return []

    hierarchical = getattr(output_dto, "hierarchical", None) or {}
    top = hierarchical.get("top")
    children = hierarchical.get("children") or {}

    messages = []
    for oid in probabilities.index:
        flat = _row(probabilities, oid)
        tree = {}
        if top is not None and oid in top.index:
            tree["top"] = _row(top, oid)
        tree["children"] = {
            name: _row(frame, oid)
            for name, frame in children.items()
            if frame is not None and oid in frame.index
        }
        features = collapsed[int(oid)].get("features") or {}
        messages.append(
            {
                "oid": str(oid),
                "features": {name: _clean(value) for name, value in features.items()},
                "lc_classification": {
                    "probabilities": flat,
                    "class": max(flat, key=flat.get),
                    "hierarchical": tree,
                },
            }
        )
    return messages
