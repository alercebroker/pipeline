"""The downstream payload is a PLACEHOLDER (design doc §9). These tests pin only
that it is well-formed and cannot throw — not that the shape is a contract."""
from types import SimpleNamespace

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
