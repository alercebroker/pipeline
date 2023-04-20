from .data.messages import data
from scripts.run_step import step_factory


def test_execute(env_variables):
    step = step_factory()
    formatted_data = step.pre_execute(data)
    result = step.execute(formatted_data)
    for d in data:
        assert d["aid"] in result
        assert "meanra" in result[d["aid"]]
        assert "meandec" in result[d["aid"]]
        assert "magstats" in result[d["aid"]]
        assert "oid" in result[d["aid"]]
        assert "tid" in result[d["aid"]]
        assert "firstmjd" in result[d["aid"]]
        assert "lastmjd" in result[d["aid"]]
        assert "ndet" in result[d["aid"]]
        assert "sigmara" in result[d["aid"]]
        assert "sigmadec" in result[d["aid"]]
