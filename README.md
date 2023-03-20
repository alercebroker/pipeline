# ALeRCE Scribe for MongoDB

This step will process the commands published in the consumer topics and perform DB operations according to their types:

## Command Format

This step expects the message to come with a single field named `payload`, whose content must be a
stringified JSON containing a single command.

The command must be formatted as follows:

```json
{
    "collection": "name",
    "type": "insert",
    "criteria": {"_id": "value", ...},
    "data": {"field": "value", ...},
    "options": {"upsert": true}
}
```
- Accepted `type` for operations are one of the following:
  * `"insert"`: The data will be inserted as a new document in the collection
  * `"update"`: The first document matching the criteria will have the data added/modified in the collection 
  * `"update_probabilities"`: Updates an existing probability, otherwise adds it. Data structure, e.g.,
    ```json
    {
        "classifier_name": "stamp_classifier",
        "classifier_version": "1.0.0",
        "SN": 0.12,
        "AGN": 0.34,
        ...
    }
    ```
- The command *must* include a collection to work with. Currently, the supported collections are:
  * `"object"`
  * `"detections"`
  * `"non_detections"`
- Except for `"insert"`, all other types require a non-empty `"criteria"` to match documents in respective database. This must include an `_id` match.
- The supported options are `"upsert"` and `"set_on_insert"`. These are ignored by the `"insert"` type.
  * `"upsert"` will add a new document with the updated data if one doesn't exist
  * `"set_on_insert"` will only add the new document (or new set of probabilities) if it doesn't exist, without modifying the existing one

## Suggested schema

For steps that sand data to the scribe, the following producer configuration is recommended, specially for the schema:

```python
SCRIBE_PRODUCER_CONFIG = {
    "TOPIC": os.environ["SCRIBE_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["SCRIBE_SERVER"],
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
