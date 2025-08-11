import sys
from typing import Any, Callable, cast

import pandas as pd
from fastavro.schema import load_schema as _load_schema

# Overwrite type as it is wrongly infered as `Unknown`
load_schema = cast(Callable[[str], Any], _load_schema)

DType = (
    pd.Int32Dtype
    | pd.Int64Dtype
    | pd.Float32Dtype
    | pd.Float64Dtype
    | pd.BooleanDtype
    | pd.StringDtype
)

type_mappings = {
    "long": "pd.Int64Dtype()",
    "int": "pd.Int32Dtype()",
    "float": "pd.Float32Dtype()",
    "double": "pd.Float64Dtype()",
    "boolean": "pd.BooleanDtype()",
    "string": "pd.StringDtype()",
}


def process_schema(path: str) -> dict[str, DType]:
    parsed_schema = load_schema(path)
    fields: list[dict[str, str | list[str] | dict[str, str]]] = parsed_schema["fields"]

    schema = {}
    for field in fields:
        field_name = field["name"]
        field_type = field["type"]

        if type(field_type) is list:
            field_type = list(filter(lambda x: x != "null", field_type))
            assert len(field_type) == 1
            field_type = field_type[0]

        if type(field_type) is dict:
            assert "logicalType" in field_type
            assert field_type["logicalType"] == "timestamp-micros"
            field_type = field_type["type"]

        assert type(field_name) is str
        assert type(field_type) is str

        field_type = type_mappings[field_type]

        schema[field_name] = field_type

    return schema


def print_schemas(pd_schemas: dict[str, dict[str, DType]]):
    print(
        """import pandas as pd


DType = (
    pd.Int32Dtype
    | pd.Int64Dtype
    | pd.Float32Dtype
    | pd.Float64Dtype
    | pd.BooleanDtype
    | pd.StringDtype
)"""
    )

    for name, schema in pd_schemas.items():
        print(f"\n{name}_schema: dict[str, DType] = {{")
        for schema_field, schema_type in schema.items():
            print(f'    "{schema_field}": {schema_type},')
        print("}")


survey_schemas = {
    "lsst": {
        "dia_forced_source": "../schemas/surveys/lsst_v8.0/lsst.v8_0.diaForcedSource.avsc",
        "dia_source": "../schemas/surveys/lsst_v8.0/lsst.v8_0.diaSource.avsc",
        "ss_source": "../schemas/surveys/lsst_v8.0/lsst.v8_0.ssSource.avsc",
        "dia_object": "../schemas/surveys/lsst_v8.0/lsst.v8_0.diaObject.avsc",
        "mpcorb": "../schemas/surveys/lsst_v8.0/lsst.v8_0.MPCORB.avsc",
    },
    "ztf": {
        "candidate": "../schemas/ztf/candidate.avsc",
        "fp_hist": "../schemas/ztf/fp_hist.avsc",
        "prv_candidate": "../schemas/ztf/prv_candidate.avsc",
    },
}


def generate(args: list[str] = sys.argv):
    assert len(args) == 2
    assert args[1] in survey_schemas.keys()
    survey = args[1]

    schemas = survey_schemas[survey]

    pd_schemas = {name: process_schema(schema) for name, schema in schemas.items()}

    print_schemas(pd_schemas)
