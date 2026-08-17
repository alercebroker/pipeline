"""The downstream payload is a PLACEHOLDER (design doc §9). These tests pin only
that it is well-formed and cannot throw — not that the shape is a contract."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lc_classification_multisurvey_step.output_parser import MultisurveyOutputParser


def frame(index, data):
    df = pd.DataFrame(data, index=index)
    df.index.name = "oid"
    return df


def make_dto(flat, top=None):
    return SimpleNamespace(
        probabilities=flat,
        hierarchical={"top": top, "children": {}},
    )


class TestMultisurveyOutputParser:
    def test_one_message_per_oid_with_top_class_per_head(self):
        dto = make_dto(
            flat=frame([1, 2], {"SNIa": [0.9, 0.1], "AGN": [0.1, 0.9]}),
            top=frame([1, 2], {"Transient": [0.8, 0.2], "Stochastic": [0.2, 0.8]}),
        )

        out = MultisurveyOutputParser().parse(dto, base_name="base", version="2.1.0").value

        assert [m["oid"] for m in out] == [1, 2]
        first = out[0]
        assert first["classifier_name"] == "base"
        assert first["classifier_version"] == "2.1.0"
        assert first["top_class"]["base"] == {"class_name": "SNIa", "probability": 0.9}
        assert first["top_class"]["base_top"]["class_name"] == "Transient"
        assert out[1]["top_class"]["base"]["class_name"] == "AGN"

    def test_missing_heads_are_absent_not_null(self):
        dto = make_dto(flat=frame([1], {"SNIa": [1.0]}), top=None)
        out = MultisurveyOutputParser().parse(dto, base_name="base", version="2.1.0").value
        assert list(out[0]["top_class"]) == ["base"]

    def test_empty_output_dto_produces_no_messages(self):
        dto = make_dto(flat=pd.DataFrame())
        assert MultisurveyOutputParser().parse(dto, base_name="base", version="2.1.0").value == []

    def test_none_output_dto_produces_no_messages(self):
        assert MultisurveyOutputParser().parse(None, base_name="base", version="2.1.0").value == []

    def test_probability_is_the_top_class_not_the_first_column(self):
        """The reported probability must belong to the winning class, not to
        whichever class happens to sit in column 0."""
        dto = make_dto(flat=frame([1], {"SNIa": [0.25], "AGN": [0.75]}))

        out = MultisurveyOutputParser().parse(dto, base_name="base", version="2.1.0").value

        assert out[0]["top_class"]["base"] == {"class_name": "AGN", "probability": 0.75}

    def test_head_covering_only_some_oids_is_absent_for_the_others(self):
        """Hierarchical heads only score the oids routed to them, so a head frame
        may be indexed on a subset of the batch."""
        dto = make_dto(
            flat=frame([1, 2], {"SNIa": [0.9, 0.1], "AGN": [0.1, 0.9]}),
            top=frame([2], {"Transient": [0.3], "Stochastic": [0.7]}),
        )

        out = MultisurveyOutputParser().parse(dto, base_name="base", version="2.1.0").value

        assert list(out[0]["top_class"]) == ["base"]
        assert list(out[1]["top_class"]) == ["base", "base_top"]
        assert out[1]["top_class"]["base_top"]["class_name"] == "Stochastic"

    def test_head_with_rows_but_no_columns_is_skipped_not_fatal(self):
        """A zero-column frame has rows, so a row-count guard lets it through and
        idxmax then blows up the whole batch (design §8: log and drop, never
        kill the batch)."""
        no_columns = MultisurveyOutputParser().parse(
            make_dto(flat=pd.DataFrame(index=[1, 2])), base_name="base", version="2.1.0"
        ).value
        assert [m["oid"] for m in no_columns] == [1, 2]
        assert all(m["top_class"] == {} for m in no_columns)

        empty_head = MultisurveyOutputParser().parse(
            make_dto(flat=frame([1], {"SNIa": [1.0]}), top=pd.DataFrame(index=[1])),
            base_name="base",
            version="2.1.0",
        ).value
        assert list(empty_head[0]["top_class"]) == ["base"]

    def test_all_nan_row_drops_that_head_for_that_oid(self):
        """An all-NaN row has no argmax. The oid keeps its message; the head is
        simply absent, as for an oid the head never scored."""
        dto = make_dto(flat=frame([1, 2], {"SNIa": [0.4, np.nan], "AGN": [0.6, np.nan]}))

        out = MultisurveyOutputParser().parse(dto, base_name="base", version="2.1.0").value

        assert [m["oid"] for m in out] == [1, 2]
        assert out[0]["top_class"]["base"] == {"class_name": "AGN", "probability": 0.6}
        assert out[1]["top_class"] == {}
