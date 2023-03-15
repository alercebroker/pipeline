# ALeRCE Scribe for MongoDB

This step will process the commands published in the consumer topics and perform DB operations according to their types:

## Command Format

The commands must be formatted as follows:

```js
{
    "collection": "name",
    "type": "insert",
    "criteria": {"field": "value", ...},
    "data": {"field": "value", ...},
    "options": {"upsert": true}
}
```
- Accepted `type` for operations are one of the following:
  * `"insert"`: The data will be inserted as a new document in the collection
  * `"update"`: The first document matching the criteria will have the data added/modified in the collection 
  * `"insert_probabilities"`: Adds a new set of probabilities in the document matching the criteria, unless it's already present. Data example:
    ```python
    {
        "classifier_name": "stamp_classifier",
        "classifier_version": "1.0.0",
        "SN": 0.12,
        "AGN": 0.34,
        ...: ...  # Remaining classes
    }
    ```
  * `"update_probabilities"`: Updates an existing probability, otherwise adds it. Data has the same structure as that for `insert_probabilities`
- The command *must* include a collection to work with. Currently, the supported collections are:
  * `"object"`
  * `"detections"`
  * `"non_detections"`
- Except for `"insert"`, all other types require a non-empty `"criteria"` to match documents in respective database. This must include an `_id` match.
- The only supported option at the time is `"upsert"`. This is ignored by the `"insert"` type.

## Suggested schema

```python
PRODUCER_CONFIG = {
    "TOPIC": "test_topic",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092"
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
