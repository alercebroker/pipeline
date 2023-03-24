from data.messages import data
from scripts.run_step import step_factory


def test_execute():
    step = step_factory()
    result = step.execute(data)
    for i, d in enumerate(data):
        assert "aid" in result[i] and d["aid"] == result[i]["aid"]
        assert "meanra" in result[i]
        assert "meandec" in result[i]
        assert "magstats" in result[i]
        assert "oid" in result[i]
        assert "tid" in result[i]
        assert "firstmjd" in result[i]
        assert "lastmjd" in result[i]
        assert "ndet" in result[i]
        assert "sigmara" in result[i]
        assert "sigmadec" in result[i]
