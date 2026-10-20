"""Reclassify from stored features: <schema>.feature rows -> model frame -> per-oid output.

The batch runner (scripts/offline_run_batch.py) computes features from the light
curve and then classifies. This module is the half that skips the computation:
it takes the rows the feature table already holds for a set of oids, rebuilds
the wide frame the BHRF model consumes, classifies the whole minibatch in one
call, and hands back one single-oid OutputDTO per object so
probability_writer.build_probability_rows works unchanged.

Two things make the rebuild non-trivial and are tested on their own:

  * feature_name_lut spells colours with hyphens and ratios with slashes; the
    model's feature_list uses underscores, and the (g,r) pair is band 12.
    Getting this wrong is silent: ~31 of the 199 features land as NaN and the
    model returns plausible but different probabilities.
  * the feature upsert key (oid, sid, feature_id, band) carries no run id, so a
    feature computed in an older pass but not in the newest keeps its stale row.
    The classifier that produced the stored probabilities saw NaN for it, so
    only the rows of the newest updated_date per oid are kept.
"""
import pandas as pd  # frames in, frames out

from alerce_classifiers.base.dto import OutputDTO  # the model's output container
from alerce_classifiers.base.factories import input_dto_factory  # builds the features-only DTO

ROW_COLUMNS = ["oid", "feature_name", "band", "value", "updated_date"]  # what fetch_feature_rows returns


def model_column(lut_name: str, band: int) -> str:  # pure; tested on its own
    """feature_name_lut spelling + band -> the column name in the model's feature_list.

    `W1-W2` -> `W1_W2`, `Power_rate_1/2` at band 12 -> `Power_rate_1_2_12`; band 0
    is band-agnostic and gets no suffix.
    """
    name = lut_name.replace("-", "_").replace("/", "_")  # the model uses underscores throughout
    return f"{name}_{band}" if band else name  # 1 = g, 2 = r, 12 = (g,r); 0 = no band


def current_rows(rows: pd.DataFrame) -> pd.DataFrame:  # rows as fetch_feature_rows returns them
    """Keep, per oid, only the rows dated with that oid's newest updated_date.

    Rows of an older computation that the newest one did not overwrite are
    dropped, because the stored probabilities were computed without them. A row
    with a NULL date next to dated siblings is of unknown age and is dropped
    too; an oid with no dated row at all keeps everything it has.
    """
    if len(rows) == 0:  # nothing to filter; keep the columns
        return rows
    dated = rows.dropna(subset=["updated_date"])  # NULL dates cannot take part in the max
    per_oid = dated.groupby("oid")["updated_date"].max()  # newest date of each oid that has one
    newest = rows["oid"].map(per_oid)  # aligned to rows; NaN for an oid with no dated row
    undated_oid = newest.isna()  # True on every row of such an oid
    return rows[(rows["updated_date"] == newest) | undated_oid]  # newest pass, or all rows when none is dated


def features_frame(rows: pd.DataFrame, columns: list) -> pd.DataFrame:  # columns = the model's feature_list
    """Pivot feature rows into one row per oid with exactly the model's columns.

    `columns` is the model's feature_list. A feature with no row is NaN (the
    model's preprocessor maps it to -999 like production); a row whose name is
    not in the list is dropped. Raises if the same (oid, column) appears twice,
    which would mean two LUT names collapsed onto one model column.
    """
    if len(rows) == 0:  # an empty minibatch still needs the model's columns
        return pd.DataFrame(columns=columns, dtype="float64")
    named = rows.assign(column=[model_column(n, int(b))  # translate every row's name once
                                for n, b in zip(rows["feature_name"], rows["band"])])
    wide = named.pivot(index="oid", columns="column", values="value")  # one row per oid; raises on duplicates
    wide = wide.reindex(columns=columns)  # the model's exact column set, in its order
    wide.index.name = "oid"  # the DTO convention (see classify.features_message_to_dto)
    return wide.astype("float64")  # values are reals; the preprocessor casts anyway


def classify_frame(frame: pd.DataFrame, model) -> OutputDTO:  # model = SquidwardFeaturesClassifier
    """Features-only DTO from a wide frame -> the model's OutputDTO for every row.

    Mirrors classify.classify_astro_object after the parser: empty detections,
    non-detections, xmatch and stamps; can_predict then predict. An empty
    OutputDTO when the model cannot predict (no rows).
    """
    empty = pd.DataFrame()  # the model reads features only
    dto = input_dto_factory(empty, empty, frame, empty, empty)  # same factory as the live step
    can, _ = model.can_predict(dto)  # False on an empty frame
    if not can:  # nothing to classify in this minibatch
        return OutputDTO(pd.DataFrame(), {"top": pd.DataFrame(), "children": {}})
    return model.predict(dto)  # one call for the whole minibatch


def _one_row(frame, oid: int) -> pd.DataFrame:  # helper of split_output
    """The single-row slice of `frame` for `oid`, or an empty frame when absent."""
    if frame is None or oid not in frame.index:  # the branch frame may not exist or not hold the oid
        return pd.DataFrame()
    return frame.loc[[oid]]  # list indexer keeps it a one-row DataFrame, not a Series


def split_output(dto: OutputDTO, oid: int) -> OutputDTO:  # one oid out of a batched DTO
    """The single-oid OutputDTO that build_probability_rows expects, cut from a batched one."""
    hierarchical = dto.hierarchical or {}  # a can't-predict DTO carries an empty dict
    children = hierarchical.get("children", {}) or {}  # the three branch frames
    return OutputDTO(  # same shape the model returns, one row everywhere
        _one_row(dto.probabilities, oid),  # flat 21-class row
        {"top": _one_row(hierarchical.get("top"), oid),  # top-level row
         "children": {name: _one_row(frame, oid) for name, frame in children.items()}},  # one row per branch
    )  # build_probability_rows demands one-row frames
