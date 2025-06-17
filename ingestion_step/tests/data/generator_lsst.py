from pathlib import Path
from random import choice, randint
from typing import Any, Callable, Iterable, cast

from fastavro.schema import load_schema as _load_schema
from fastavro.types import Schema
from fastavro.utils import generate_many

from ingestion_step.core.types import Message

# Overwrite type as it is wrongly infered as `Unknown`
load_schema = cast(Callable[[str], Schema], _load_schema)

base_schema_path = Path("../schemas/surveys/lsst")


def generate_bands(alert: Message) -> Message:
    alert["diaSource"]["band"] = choice("gr")
    for prv_source in alert["prvDiaSources"] or []:
        prv_source["band"] = choice("gr")
    for forced_source in alert["prvDiaForcedSources"] or []:
        forced_source["band"] = choice("gr")
    for non_detection in alert["prvDiaNondetectionLimits"] or []:
        non_detection["band"] = choice("gr")

    return alert


def generate_object_ids(alert: Message) -> Message:
    obj = "ss" if alert["diaObject"] is None else "dia"
    if obj == "dia":
        alert["diaSource"]["diaObjectId"] = randint(0, 2**63 - 1)
        alert["diaSource"]["ssObjectId"] = None
        for prv_source in alert["prvDiaSources"] or []:
            prv_source["diaObjectId"] = randint(0, 2**63 - 1)
            prv_source["ssObjectId"] = None
    else:
        alert["diaSource"]["diaObjectId"] = None
        alert["diaSource"]["ssObjectId"] = randint(0, 2**63 - 1)
        for prv_source in alert["prvDiaSources"] or []:
            prv_source["diaObjectId"] = None
            prv_source["ssObjectId"] = randint(0, 2**63 - 1)

    return alert


def generate_alerts(n: int = 30) -> Iterable[dict[str, Any]]:
    schema_path = str(base_schema_path.joinpath("v7_4_alert.avsc"))
    schema = load_schema(schema_path)
    alerts = generate_many(schema, n)

    alerts = map(generate_object_ids, alerts)
    alerts = map(generate_bands, alerts)

    return alerts
