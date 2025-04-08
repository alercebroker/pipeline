import random
import string
from pathlib import Path
from typing import Any, Callable, Iterable, cast

from fastavro.schema import load_schema as _load_schema
from fastavro.types import Schema
from fastavro.utils import generate_many
from numpy.random import randint

# Overwrite type as it is wrongly infered as `Unknown`
load_schema = cast(Callable[[str], Schema], _load_schema)

base_schema_path = Path("../schemas/ztf")


def replace_fid_det(det: dict[str, Any]) -> dict[str, Any]:
    det["fid"] = randint(1, 4)
    return det


def replace_all_fids(alert: dict[str, Any]) -> dict[str, Any]:
    alert["candidate"] = replace_fid_det(alert["candidate"])
    if alert["prv_candidates"] is not None:
        alert["prv_candidates"] = list(map(replace_fid_det, alert["prv_candidates"]))
    if alert["fp_hists"] is not None:
        alert["fp_hists"] = list(map(replace_fid_det, alert["fp_hists"]))
    return alert


def replace_objectId(alert: dict[str, Any]) -> dict[str, Any]:
    alert["objectId"] = "ZTF18" + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=7)
    )
    return alert


def generate_alerts(n: int = 10) -> Iterable[dict[str, Any]]:
    schema_path = str(base_schema_path.joinpath("alert.avsc"))
    schema = load_schema(schema_path)
    alerts = generate_many(schema, n)

    alerts = map(replace_all_fids, alerts)
    alerts = map(replace_objectId, alerts)

    return alerts
