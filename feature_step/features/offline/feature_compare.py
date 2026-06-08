"""Compare our computed features against ALeRCE-stored legacy features.

Pure module — no DB connections. Uses fid_mapper_for_db from features.utils.parsers
to convert our letter-fid convention ("g", "r", "g,r", None) to the integer
convention used in alerce.feature (1, 2, 12, 0).
"""
import re

import numpy as np
import pandas as pd
from features.utils.parsers import fid_mapper_for_db

_MAX_ONLY_NAMES = 30  # max entries in only_*_names lists in the summary


def _version_sort_key(version: str) -> tuple:
    """Numeric sort key for a version string.

    Splits on '.' and parses the leading integer of each component via regex.
    Non-numeric or missing leading digits map to 0.  E.g.:
        '27.5.7a32.dev1' -> (27, 5, 7, 0)
        '26.0.0'         -> (26, 0, 0)
        '25.0.1a13'      -> (25, 0, 1)
    Tie-break appends the raw string so the result is deterministic.
    """
    parts = version.split(".")
    nums = []
    for part in parts:
        m = re.match(r"\d+", part)
        nums.append(int(m.group()) if m else 0)
    return tuple(nums) + (version,)


def latest_feature_version(versions: list[str]) -> str | None:
    """Return the best default version from a list of alerce feature version strings.

    Logic:
    - Partition into "modern" (first char is a digit) and "legacy" (everything
      else, e.g. 'lc_classifier_*').
    - Prefer modern: if any modern versions exist, choose among them; otherwise
      fall back to legacy.
    - Within the chosen group pick the max by numeric sort key (leading integer
      of each dot-separated component; non-numeric parts → 0).  Tie-break by
      raw string for determinism.
    - Returns None for an empty input list.
    """
    if not versions:
        return None
    modern = [v for v in versions if v and v[0].isdigit()]
    pool = modern if modern else versions
    return max(pool, key=_version_sort_key)


def _fid_letters_to_int(fid_series: pd.Series) -> pd.Series:
    """Map a Series of fid letters/None to the alerce integer fid convention."""
    return fid_series.map(fid_mapper_for_db)


def compare_feature_frames(
    ours: pd.DataFrame,
    theirs: pd.DataFrame,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> tuple[pd.DataFrame, dict]:
    """Align and diff our features against ALeRCE-stored features.

    Args:
        ours: DataFrame with columns [name, value, fid] where fid is a letter
              string ("g", "r", "g,r") or None.
        theirs: DataFrame with columns [name, value, fid] where fid is an
                integer (1, 2, 12, 0).
        rtol: Relative tolerance for numpy.isclose match check.
        atol: Absolute tolerance for numpy.isclose match check.

    Returns:
        (merged, summary) where:
        - merged: one row per (name, fid_int) union, with columns:
            name, fid_int, value_ours, value_theirs, abs_diff, rel_diff, status
          status in {"match", "differ", "only_ours", "only_theirs"}.
          NaN == NaN counts as "match".
        - summary: dict with keys:
            match, differ, only_ours, only_theirs,
            n_compared, n_match,
            only_ours_names (sorted list, up to _MAX_ONLY_NAMES),
            only_theirs_names (sorted list, up to _MAX_ONLY_NAMES).
    """
    # --- Normalize our fid letters to int ---
    ours_norm = ours[["name", "value", "fid"]].copy()
    ours_norm["fid_int"] = _fid_letters_to_int(ours_norm["fid"])
    ours_norm = ours_norm[["name", "fid_int", "value"]].rename(columns={"value": "value_ours"})

    theirs_norm = theirs[["name", "value", "fid"]].copy()
    theirs_norm = theirs_norm.rename(columns={"fid": "fid_int", "value": "value_theirs"})

    # --- Outer join on (name, fid_int) ---
    merged = pd.merge(
        ours_norm,
        theirs_norm,
        on=["name", "fid_int"],
        how="outer",
    )

    # --- Compute abs_diff and rel_diff ---
    # Use float NaN as sentinel for rows without both values
    merged["abs_diff"] = np.nan
    merged["rel_diff"] = np.nan

    mask_both = merged["value_ours"].notna() & merged["value_theirs"].notna()
    mask_both_nan = merged["value_ours"].isna() & merged["value_theirs"].isna()

    # Numeric abs/rel diff only when both sides have finite values
    if mask_both.any():
        a = merged.loc[mask_both, "value_ours"].to_numpy(dtype=float)
        b = merged.loc[mask_both, "value_theirs"].to_numpy(dtype=float)
        abs_d = np.abs(a - b)
        # rel_diff = abs_diff / max(|b|, atol) — avoids division by zero
        rel_d = abs_d / np.maximum(np.abs(b), atol)
        merged.loc[mask_both, "abs_diff"] = abs_d
        merged.loc[mask_both, "rel_diff"] = rel_d

    # --- Assign status ---
    # Identify which rows came from only one side using the outer join NaN pattern.
    # After outer join: a row is "only_ours" if it was absent from theirs (value_theirs is NaN
    # AND no theirs entry matched). We use indicator-style detection.
    merged_indicator = pd.merge(
        ours_norm,
        theirs_norm,
        on=["name", "fid_int"],
        how="outer",
        indicator=True,
    )
    merged["_indicator"] = merged_indicator["_merge"]

    def _status_with_indicator(row):
        ind = row["_indicator"]
        if ind == "left_only":
            return "only_ours"
        if ind == "right_only":
            return "only_theirs"
        # Both present
        v_ours = row["value_ours"]
        v_theirs = row["value_theirs"]
        both_nan = pd.isna(v_ours) and pd.isna(v_theirs)
        if both_nan:
            return "match"
        if pd.isna(v_ours) or pd.isna(v_theirs):
            # One is NaN but not the other
            return "differ"
        if np.isclose(float(v_ours), float(v_theirs), rtol=rtol, atol=atol):
            return "match"
        return "differ"

    merged["status"] = merged.apply(_status_with_indicator, axis=1)
    merged = merged.drop(columns=["_indicator"])

    # --- Build summary ---
    status_counts = merged["status"].value_counts().to_dict()
    for s in ("match", "differ", "only_ours", "only_theirs"):
        status_counts.setdefault(s, 0)

    n_compared = status_counts["match"] + status_counts["differ"]
    n_match = status_counts["match"]

    only_ours_names = sorted(
        merged.loc[merged["status"] == "only_ours", "name"].unique().tolist()
    )[:_MAX_ONLY_NAMES]
    only_theirs_names = sorted(
        merged.loc[merged["status"] == "only_theirs", "name"].unique().tolist()
    )[:_MAX_ONLY_NAMES]

    summary = {
        "match": status_counts["match"],
        "differ": status_counts["differ"],
        "only_ours": status_counts["only_ours"],
        "only_theirs": status_counts["only_theirs"],
        "n_compared": n_compared,
        "n_match": n_match,
        "only_ours_names": only_ours_names,
        "only_theirs_names": only_theirs_names,
    }

    return merged, summary
