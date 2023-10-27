from fastavro.schema import load_schema
from fastavro.utils import generate_one


def get_content(schema_path):
    schema = load_schema(schema_path)
    generator = generate_one(schema)
    return generator
