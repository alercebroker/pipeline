import pandas as pd
from magstats_step.core.utils.create_dataframe import generate_non_detections_dataframe
from data.messages import data


def test_generate_non_detections_df():
    non_det_df = generate_non_detections_dataframe(data[0]["non_detections"])
    assert isinstance(non_det_df, pd.DataFrame)
    assert len(non_det_df) > 0


def test_generate_non_detectons_empty():
    non_det_df = generate_non_detections_dataframe([])
    assert isinstance(non_det_df, pd.DataFrame)
    assert non_det_df.shape[0] == 0
