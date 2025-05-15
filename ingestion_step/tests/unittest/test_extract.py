# pyright: reportPrivateUsage=false
from typing import Any

import fastavro
import pandas as pd

import ingestion_step.ztf.extractor as extractor


def test_extract_candidates(ztf_alerts: list[dict[str, Any]]):
    candidates = extractor._extract_candidates(ztf_alerts)
    for col in candidates.values():
        assert len(ztf_alerts) == len(col)

    fields = {"objectId", "candid", "parent_candid", "has_stamp"}
    assert fields <= set(candidates.keys())


def test_extract_prv_candidates(ztf_alerts: list[dict[str, Any]]):
    prv_candidates = extractor._extract_prv_candidates(ztf_alerts)

    fields = {"objectId", "parent_candid", "has_stamp"}
    assert fields <= set(prv_candidates.keys())


def test_extract_fp_hists(ztf_alerts: list[dict[str, Any]]):
    fp_hists = extractor._extract_fp_hists(ztf_alerts)

    fields = {
        "objectId",
        "ra",
        "dec",
        "magzpsci",
        "forcediffimflux",
        "forcediffimfluxunc",
        "parent_candid",
        "has_stamp",
    }

    assert fields <= set(fp_hists.keys())


def test_extract(ztf_alerts: list[dict[str, Any]]):
    ztf_data = extractor.extract(ztf_alerts)

    n_candidates = len(ztf_alerts)
    n_prv_candidates = sum(
        len(alert["prv_candidates"]) for alert in ztf_alerts if alert["prv_candidates"]
    )
    n_fp_hists = sum(
        len(alert["fp_hists"]) for alert in ztf_alerts if alert["fp_hists"]
    )

    assert n_candidates == len(ztf_data["candidates"])
    assert n_prv_candidates == len(ztf_data["prv_candidates"])
    assert n_fp_hists == len(ztf_data["fp_hists"])


def test_candid_precision():
    with open("tests/data/avros/3044341005815015012.avro", "rb") as file:
        reader = fastavro.reader(file)
        messages = [record for record in reader]

    prv_candidates = extractor._extract_prv_candidates(messages)
    prv_pandas = pd.DataFrame(prv_candidates)
    expected = [
        3017328595815015006,
        None,
        3019328815815015006,
        None,
        None,
        None,
        None,
        None,
        3036386425815015013,
        None,
        None,
    ]

    for i, value in enumerate(prv_pandas["candid"]):
        assert value == expected[i]
