# ALeRCE Scribe for MongoDB

This step will process the commands published in the Command topic and perform DB operations in regard to the commands.

## Command Format

The commands must be formatted as follows:

```js
{
    "collection": "The collection that will be written on",
    "type": "insert" | "update",
    "criteria": "JSON or dictionary which represents the filter of the query",
    "data": "JSON or dictionary which represents the data to be inserted or updated",
    "options": "(Optional) JSON or dictionary which represents supported DB options"
}
```
 - The command must include a collection to work with. Currently the supported collections are ``"object"``, ``"detections"`` and ``"non_detections"``.
 - To be published to the topic, this JSON or dictionary must be stringified (*tested with json.dumps*) and published in a dictionary with the key **payload** and serialized as AVRO.
 - An update command must include a non-null criteria.
 - Criteria and data haven't been tested with MongoDB operations.
 - 

 ## Suggested schema
 ```python
 PRODUCER_CONFIG = {
    "TOPIC": "test_topic",
    "PARAMS": {
        "bootstrap.servers": "localhost:9094"
    },
    "SCHEMA": {
        "namespace": "db_operation",
        "type": "record",
        "name": "Command",
        "fields": [
            {"name": "payload", "type": "string"},
        ],
    },
}
 ```