from typing import Any, Callable, cast

import pandas as pd
from fastavro.schema import load_schema as _load_schema

# Overwrite type as it is wrongly infered as `Unknown`
load_schema = cast(Callable[[str], Any], _load_schema)

DTypes = (
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


def process_schema(path: str) -> dict[str, DTypes]:
    parsed_schema = load_schema(path)
    fields: list[dict[str, str | list[str]]] = parsed_schema["fields"]

    schema = {}
    for field in fields:
        field_name = field["name"]
        field_type = field["type"]

        if type(field_type) is list:
            field_type = list(filter(lambda x: x != "null", field_type))
            assert len(field_type) == 1
            field_type = field_type[0]

        assert type(field_name) is str
        assert type(field_type) is str

        field_type = type_mappings[field_type]

        schema[field_name] = field_type

    return schema


def print_schemas(pd_schemas: dict[str, dict[str, DTypes]]):
    print("import pandas as pd")
    for name, schema in pd_schemas.items():
        print()
        print(f"{name}_schema = {{")
        for schema_field, schema_type in schema.items():
            print(f'    "{schema_field}": {schema_type},')
        print("}")


def generate_lsst():
    base_path = "../schemas/surveys/lsst/"
    schemas = {
        "dia_forced_source": "v7_4_diaForcedSource.avsc",
        "dia_non_detection_limit": "v7_4_diaNondetectionLimit.avsc",
        "dia_source": "v7_4_diaSource.avsc",
        "dia_object": "v7_4_diaObject.avsc",
        "ss_object": "v7_4_ssObject.avsc",
    }

    pd_schemas = {
        name: process_schema(base_path + schema)
        for name, schema in schemas.items()
    }

    print_schemas(pd_schemas)


def generate_ztf():
    base_path = "../schemas/ztf/"
    schemas = {
        "candidate": "candidate.avsc",
        "fp_hist": "fp_hist.avsc",
        "prv_candidate": "prv_candidate.avsc",
    }

    pd_schemas = {
        name: process_schema(base_path + schema)
        for name, schema in schemas.items()
    }

    print_schemas(pd_schemas)
