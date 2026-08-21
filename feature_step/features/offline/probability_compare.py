"""Compare our offline BHRF probabilities against ALeRCE-stored legacy probabilities.

Pure module — no DB connections (sibling of feature_compare.py). The offline side
is the BHRF OutputDTO (5 hierarchical heads); the stored side is rows read from
alerce.probability. Both carry the same string `class_name`s (same model), so the
comparison is a straight (classifier_name, class_name) join — no class_id mapping.

The classifier_id -> classifier_name map (== alerce.probability.classifier_name)
and the OutputDTO -> head-frame mapping are reused from classifier_taxonomy_lut and
probability_writer so this stays the single comparison, not a second copy of the
head wiring.
"""
import numpy as np
import pandas as pd

from features.offline.classifier_taxonomy_lut import CLASSIFIER_LUT
from features.offline.probability_writer import _iter_frames

# classifier_id -> classifier_name; the names equal alerce.probability.classifier_name.
CLASSIFIER_NAME_BY_ID = {d["classifier_id"]: d["classifier_name"] for d in CLASSIFIER_LUT}
# The 5 BHRF head names, in id order (flat, top, transient, stochastic, periodic).
BHRF_CLASSIFIER_NAMES = [CLASSIFIER_NAME_BY_ID[i] for i in sorted(CLASSIFIER_NAME_BY_ID)]


def offline_dto_to_series(output_dto) -> dict:
    """BHRF OutputDTO -> {classifier_name: Series(class_name -> probability)}.

    Skips heads the model didn't emit (empty/None frame). Uses the same
    OutputDTO->head mapping as the probability writer (single source of truth).
    """
    out = {}
    for classifier_id, frame in _iter_frames(output_dto):
        if frame is None or len(frame) == 0:
            continue
        out[CLASSIFIER_NAME_BY_ID[classifier_id]] = frame.iloc[0]
    return out


def _series_to_long(offline_by_name: dict) -> pd.DataFrame:
    """{classifier_name: Series} -> long df [classifier_name, class_name, prob_offline]."""
    rows = []
    for cname, series in offline_by_name.items():
        for class_name, prob in series.items():
            rows.append((cname, str(class_name), float(prob)))
    return pd.DataFrame(rows, columns=["classifier_name", "class_name", "prob_offline"])


def compare_probability_frames(
    offline_by_name: dict,
    stored: pd.DataFrame,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> tuple[pd.DataFrame, dict]:
    """Align and diff offline vs stored BHRF probabilities.

    Args:
        offline_by_name: {classifier_name: Series(class_name -> probability)} —
            see offline_dto_to_series.
        stored: DataFrame with columns [classifier_name, class_name, probability,
            ranking] (db.fetch_stored_probabilities output).
        rtol, atol: tolerances for numpy.isclose per-class probability match.

    Returns:
        (merged, summary).
        merged: one row per (classifier_name, class_name) union, columns
            [classifier_name, class_name, prob_offline, prob_stored, abs_diff,
             rel_diff, status]; status in
            {match, differ, only_offline, only_stored}.
        summary: dict with match/differ/only_offline/only_stored counts,
            n_compared, n_match, per-head rank-1 agreement (rank1 dict +
            rank1_agree / rank1_total), and `passed` (all compared classes match,
            no only_* rows, all rank-1 agree).
    """
    offline_long = _series_to_long(offline_by_name)

    stored_long = (
        stored[["classifier_name", "class_name", "probability"]]
        .rename(columns={"probability": "prob_stored"})
        .copy()
    )
    stored_long["class_name"] = stored_long["class_name"].astype(str)
    stored_long["prob_stored"] = stored_long["prob_stored"].astype(float)

    merged = pd.merge(
        offline_long, stored_long,
        on=["classifier_name", "class_name"], how="outer", indicator=True,
    )

    both = merged["prob_offline"].notna() & merged["prob_stored"].notna()
    merged["abs_diff"] = np.nan
    merged["rel_diff"] = np.nan
    if both.any():
        a = merged.loc[both, "prob_offline"].to_numpy(dtype=float)
        b = merged.loc[both, "prob_stored"].to_numpy(dtype=float)
        merged.loc[both, "abs_diff"] = np.abs(a - b)
        merged.loc[both, "rel_diff"] = np.abs(a - b) / np.maximum(np.abs(b), atol)

    def _status(row):
        if row["_merge"] == "left_only":
            return "only_offline"
        if row["_merge"] == "right_only":
            return "only_stored"
        if np.isclose(float(row["prob_offline"]), float(row["prob_stored"]),
                      rtol=rtol, atol=atol):
            return "match"
        return "differ"

    merged["status"] = merged.apply(_status, axis=1)
    merged = merged.drop(columns=["_merge"])

    counts = merged["status"].value_counts().to_dict()
    for s in ("match", "differ", "only_offline", "only_stored"):
        counts.setdefault(s, 0)

    # --- per-head rank-1 (argmax) agreement ---
    rank1 = {}
    stored_r1 = stored[stored["ranking"] == 1] if "ranking" in stored.columns else stored.iloc[0:0]
    for cname, series in offline_by_name.items():
        off_top = str(series.astype(float).idxmax()) if len(series) else None
        s_rows = stored_r1[stored_r1["classifier_name"] == cname]
        sto_top = str(s_rows.iloc[0]["class_name"]) if len(s_rows) else None
        rank1[cname] = {
            "offline": off_top,
            "stored": sto_top,
            "agree": off_top is not None and off_top == sto_top,
        }
    rank1_total = len(rank1)
    rank1_agree = sum(1 for v in rank1.values() if v["agree"])

    n_compared = counts["match"] + counts["differ"]
    passed = (
        counts["differ"] == 0
        and counts["only_offline"] == 0
        and counts["only_stored"] == 0
        and n_compared > 0
        and rank1_agree == rank1_total
    )

    summary = {
        "match": counts["match"],
        "differ": counts["differ"],
        "only_offline": counts["only_offline"],
        "only_stored": counts["only_stored"],
        "n_compared": n_compared,
        "n_match": counts["match"],
        "rank1": rank1,
        "rank1_agree": rank1_agree,
        "rank1_total": rank1_total,
        "passed": passed,
    }
    return merged, summary


def _rank1(long: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """One row per (oid, classifier_name): the top-probability class.

    `ranking` is ignored on purpose. The two tables were written years apart by
    different code, and a ranking convention that differs between them would
    silently decide the comparison. Ties break on class_name ascending so the
    same tie resolves the same way on both sides regardless of row order.
    """
    ordered = long.sort_values(
        ["oid", "classifier_name", "probability", "class_name"],
        ascending=[True, True, False, True], kind="mergesort",
    )
    top = ordered.groupby(["oid", "classifier_name"], as_index=False).first()
    return top.rename(columns={"class_name": f"class_{prob_col}",
                               "probability": f"prob_{prob_col}"})[
        ["oid", "classifier_name", f"class_{prob_col}", f"prob_{prob_col}"]
    ]


def rank1_agreement(ours: pd.DataFrame, legacy: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rank-1 agreement between two stored probability tables, over many oids.

    Both frames are long rows [oid, classifier_name, class_name, probability,
    ranking] -- ours read from <schema>.probability, legacy from
    alerce.probability, already translated to the shared string vocabulary.

    Only (oid, classifier_name) pairs present on BOTH sides are scored: legacy
    has no row for objects it never classified, and counting those as
    disagreements would report a defect where there is nothing to compare.

    Returns (per_oid, summary). per_oid has one row per scored pair with columns
    [oid, classifier_name, class_ours, class_legacy, prob_ours, prob_legacy,
    agree]. summary carries per-classifier {n_both, n_agree, rate} plus the
    unmatched counts.
    """
    a = _rank1(ours, "ours")
    b = _rank1(legacy, "legacy")
    merged = pd.merge(a, b, on=["oid", "classifier_name"], how="outer", indicator=True)

    both = merged[merged["_merge"] == "both"].drop(columns=["_merge"]).reset_index(drop=True)
    both["agree"] = both["class_ours"] == both["class_legacy"]

    by_classifier = {}
    for cname, grp in both.groupby("classifier_name"):
        n_both = int(len(grp))
        n_agree = int(grp["agree"].sum())
        by_classifier[str(cname)] = {
            "n_both": n_both, "n_agree": n_agree,
            "rate": (n_agree / n_both) if n_both else 0.0,
        }

    summary = {
        "by_classifier": by_classifier,
        "n_scored": int(len(both)),
        "n_only_ours": int((merged["_merge"] == "left_only").sum()),
        "n_only_legacy": int((merged["_merge"] == "right_only").sum()),
    }
    return both, summary
