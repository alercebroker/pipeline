# ALeRCE Scribe for MongoDB

This step will process the commands published in the Command topic and perform DB operations in regard to the commands.

## Command Format

The commands must be formatted as follows:

```js
{
    "type": "insert" | "update",
    "criteria": "JSON or dictionary which represent the filter of the query",
    "data": "JSON or dictionary which represent the data to be inserted or updated"
}
```

 - To be published to the topic, this JSON or dictionary must be stringified (*tested with json.dumps*) and published in a dictionary with the key **payload** and serialized as AVRO.
 - An insert command must include a non-null criteria 