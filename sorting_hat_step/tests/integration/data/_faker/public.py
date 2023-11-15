import io
import json
from fastavro import schema, writer
from confluent_kafka import Producer

mock_stamp = {"fileName": "mock", "stampData": b"bainari"}

print("Loading schema...")
ztf_schema = schema.load_schema("schema/alert.avsc")
print("Loaded!")

producer = Producer({"bootstrap.servers": "kafka:9092"})

print("Loading messages...")
f = open("batch.json", "r")
messages = json.load(f)

for m in messages:
    print(f"Publishing message with candid {m['candid']}")
    m["candidate"]["candid"] = m["candid"]
    m["cutoutScience"] = mock_stamp
    m["cutoutTemplate"] = mock_stamp
    m["cutoutDifference"] = mock_stamp
    serialized = io.BytesIO()
    writer(serialized, ztf_schema, [m])
    #serialized.seek(0)
    producer.produce("ztf", value=serialized.getvalue())
    print("Published!")
    producer.flush()

f.close()
