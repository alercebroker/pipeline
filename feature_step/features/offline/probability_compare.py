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


def _ranked(long: pd.DataFrame, side: str) -> pd.DataFrame:
    """Add a per-(oid, classifier) dense rank by probability descending.

    Ties break on class_name so both sides order an ambiguous pair the same way
    (see _rank1); otherwise the rank a class gets would depend on row order.
    """
    df = long.sort_values(["oid", "classifier_name", "probability", "class_name"],
                          ascending=[True, True, False, True], kind="mergesort").copy()
    df[f"rank_{side}"] = df.groupby(["oid", "classifier_name"]).cumcount() + 1
    return df.rename(columns={"probability": f"prob_{side}"})


def borderline_report(ours: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    """Per (oid, classifier): how decided each side was, and where the other
    side's winner sat in this side's ranking.

    Both frames are long rows [oid, classifier_name, class_name, probability]
    covering ALL classes, not just rank 1 -- the margin cannot be recovered from
    the winner alone.

    A rank-1 flip across a 0.002 gap and one across a 0.85 gap are not the same
    event; `margin_*` separates them. `rank_of_ours_in_legacy` answers the
    question a flip actually raises: was our class legacy's close second, or
    something it had ranked last? A class the other side never scored yields NaN
    rather than 0.0 -- "never considered" is a different claim from "ruled out".

    Returns one row per (oid, classifier_name) with columns [class_ours,
    class_legacy, prob_ours, prob_legacy, margin_ours, margin_legacy,
    rank_of_ours_in_legacy, prob_legacy_for_our_class, rank_of_legacy_in_ours,
    prob_ours_for_legacy_class, agree].
    """
    ro, rl = _ranked(ours, "ours"), _ranked(legacy, "legacy")

    def _margin(df, side):
        top2 = df[df[f"rank_{side}"] <= 2]
        p = top2.pivot_table(index=["oid", "classifier_name"], columns=f"rank_{side}",
                             values=f"prob_{side}", aggfunc="first")
        # A single-class head has no second place; the winner is unopposed.
        second = p[2] if 2 in p.columns else 0.0
        return (p[1] - second).rename(f"margin_{side}")

    win_o = ro[ro["rank_ours"] == 1].set_index(["oid", "classifier_name"])
    win_l = rl[rl["rank_legacy"] == 1].set_index(["oid", "classifier_name"])

    out = pd.DataFrame({
        "class_ours": win_o["class_name"], "prob_ours": win_o["prob_ours"],
    }).join(pd.DataFrame({
        "class_legacy": win_l["class_name"], "prob_legacy": win_l["prob_legacy"],
    }), how="inner")
    out = out.join(_margin(ro, "ours")).join(_margin(rl, "legacy"))

    # Locate each side's winner in the other side's ranking.
    li = rl.set_index(["oid", "classifier_name", "class_name"])[["rank_legacy", "prob_legacy"]]
    oi = ro.set_index(["oid", "classifier_name", "class_name"])[["rank_ours", "prob_ours"]]
    key_o = pd.MultiIndex.from_arrays(
        [out.index.get_level_values(0), out.index.get_level_values(1), out["class_ours"]])
    key_l = pd.MultiIndex.from_arrays(
        [out.index.get_level_values(0), out.index.get_level_values(1), out["class_legacy"]])
    out["rank_of_ours_in_legacy"] = li["rank_legacy"].reindex(key_o).to_numpy()
    out["prob_legacy_for_our_class"] = li["prob_legacy"].reindex(key_o).to_numpy()
    out["rank_of_legacy_in_ours"] = oi["rank_ours"].reindex(key_l).to_numpy()
    out["prob_ours_for_legacy_class"] = oi["prob_ours"].reindex(key_l).to_numpy()

    out["agree"] = out["class_ours"] == out["class_legacy"]
    return out.reset_index()
