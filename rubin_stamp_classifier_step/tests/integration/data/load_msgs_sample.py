## This script reads the avro files downloaded by download_msgs_sample.py

import os
import io
from struct import unpack
from fastavro import schemaless_reader
from fastavro.schema import load_schema


def load_sample_messages() -> list[dict]:
    # Get absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up four levels to reach the project root
    project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
    schema_path = os.path.join(
        project_root, "schemas", "surveys", "lsst", "v7_4_alert.avsc"
    )
    schema = load_schema(schema_path)

    # Loop over all avro files in the directory in alphabetical order
    sample_messages = []
    avro_dir = os.path.join(script_dir, "avro_messages")
    for cont,filename in enumerate(sorted(os.listdir(avro_dir))):
        if filename.endswith(".avro"):
            with open(os.path.join(avro_dir, filename), "rb") as f:
                bytes_io = io.BytesIO(f.read())
                magic, schema_id = unpack(">bI", bytes_io.read(5))
                assert schema_id == 704

                content = schemaless_reader(bytes_io, schema)
                content["diaSource"]['diaObjectId'] = cont%3
                sample_messages.append(content)
    return sample_messages


if __name__ == "__main__":
    # Get list of sample messages
    msgs = load_sample_messages()
    # Print the keys of each message
    for i, msg in enumerate(msgs):
        print(f"Message {i} keys: {msg.keys()}")
