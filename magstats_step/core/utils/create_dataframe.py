from typing import List
import pandas as pd


def generic_dataframe_from_detections(detections: List[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        detections, exclude=["extra_fields"], index="candid"
    ).sort_values("mjd")


def generic_dataframe_from_non_detections(nd: List[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(nd).sort_values("mjd")


def extra_dataframe_from_detections(detections: List[dict]) -> pd.DataFrame:
    extra_fields_list = ["distpsnr1", "sgscore1", "chinr", "sharpnr"]
    chosen_fields = {
        detection["candid"]: detection["extra_field"]
        for detection in detections
        if detection["tid"].lower().startswith("ztf")
    }
    return pd.DataFrame(
        chosen_fields.values(), index=chosen_fields.keys(), columns=extra_fields_list
    )
