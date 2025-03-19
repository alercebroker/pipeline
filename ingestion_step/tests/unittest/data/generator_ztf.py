from pathlib import Path
from typing import Any, Callable, Iterable, cast

from fastavro.schema import load_schema as _load_schema
from fastavro.types import Schema
from fastavro.utils import generate_many

# Overwrite type as it is wrongly infered as `Unknown`
load_schema = cast(Callable[[str], Schema], _load_schema)

base_schema_path = Path("../schemas/ztf")


def generate_alerts(n: int = 10) -> Iterable[dict[str, Any]]:
    schema_path = str(base_schema_path.joinpath("alert.avsc"))
    schema = load_schema(schema_path)
    alerts = generate_many(schema, n)

    return alerts
