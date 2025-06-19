from pathlib import Path
from random import choice, randint
from typing import Callable, Iterable, cast

from fastavro.schema import load_schema as _load_schema
from fastavro.types import Schema
from fastavro.utils import generate_many, generate_one

from ingestion_step.core.types import Message

# Overwrite type as it is wrongly infered as `Unknown`
load_schema = cast(Callable[[str], Schema], _load_schema)

base_schema_path = Path("../schemas/surveys/lsst/")
alert_schema = load_schema(str(base_schema_path / "v7_4_alert.avsc"))
diaObject_schema = load_schema(str(base_schema_path / "v7_4_diaObject.avsc"))
ssObject_schema = load_schema(str(base_schema_path / "v7_4_ssObject.avsc"))


def generate_bands(alert: Message) -> Message:
    alert["diaSource"]["band"] = choice("gr")
    for prv_source in alert["prvDiaSources"] or []:
        prv_source["band"] = choice("gr")
    for forced_source in alert["prvDiaForcedSources"] or []:
        forced_source["band"] = choice("gr")
    for non_detection in alert["prvDiaNondetectionLimits"] or []:
        non_detection["band"] = choice("gr")

    return alert


def generate_objects(alert: Message) -> Message:
    obj = choice(["dia", "ss"])

    if obj == "dia":
        diaObjectId = randint(0, 2**63 - 1)
        ssObjectId = None
        alert["diaObject"] = generate_one(diaObject_schema)
        alert["diaObject"]["diaObjectId"] = diaObjectId
        alert["ssObject"] = None
    else:
        diaObjectId = None
        ssObjectId = randint(0, 2**63 - 1)
        alert["diaObject"] = None
        alert["ssObject"] = generate_one(ssObject_schema)
        alert["ssObject"]["ssObjectId"] = ssObjectId

    alert["diaSource"]["diaObjectId"] = diaObjectId
    alert["diaSource"]["ssObjectId"] = ssObjectId
    for prv_source in alert["prvDiaSources"] or []:
        prv_source["diaObjectId"] = diaObjectId
        prv_source["ssObjectId"] = ssObjectId

    if obj == "ss":
        alert["prvDiaForcedSources"] = None
    for forced_source in alert["prvDiaForcedSources"] or []:
        forced_source["diaObjectId"] = diaObjectId

    return alert


def generate_alerts(n: int = 30) -> Iterable[Message]:
    alerts = generate_many(alert_schema, n)

    alerts = map(generate_objects, alerts)
    alerts = map(generate_bands, alerts)

    return alerts
