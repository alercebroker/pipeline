from fastavro import schemaless_writer
from fastavro.utils import generate_one
from fastavro.schema import load_schema
from fastavro.repository.base import SchemaRepositoryError
import io


def generate_schemaless_batch(n):
    try:
        schema = load_schema("schemas/elasticc/elasticc.v0_9_1.alert.avsc")
    except SchemaRepositoryError:
        schema = load_schema(
            "sorting_hat_step/schemas/elasticc/elasticc.v0_9_1.alert.avsc"
        )

    messages = []
    for i in range(n):
        out = io.BytesIO()
        schemaless_writer(out, schema, generate_one(schema))
        messages.append(out.getvalue())
    return messages
