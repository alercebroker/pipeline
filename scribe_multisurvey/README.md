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
    "criteria": {"field": "value", ...},
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
  * `"update_features"`: Updates an existing feature, otherwise adds it. Data structure, e.g.,
    ```json
    {
        "features_version": "lc_classifier_1.2.1-P",
        "features_group": "ztf_features"
        "features": [
            {
                "name": "Amplitude",
                "value": 0.6939550000000008,
                "fid": 0
            },
            {
                "name": "positive_fraction",
                "value": 1,
                "fid": 2
            }, 
            ...
        ]
    }
    ```
- The command *must* include a collection to work with. Currently, the supported collections are:
  * `"object"`
  * `"detections"`
  * `"non_detections"`
  * `"score"`
- Except for `"insert"`, all other types require a non-empty `"criteria"` to match documents in the respective collection.
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
