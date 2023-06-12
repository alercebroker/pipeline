CLASSIFICATIONS = {
    "type": "array",
    "items": {
        "type": "record",
        "name": "classificationDict",
        "fields": [
            # {"name": "classifier_name", "type": "string"},
            # {"name": "model_version", "type": "string"},
            {"name": "class_name", "type": "string"},
            {"name": "probability", "type": "double"},
        ],
    },
}

SCHEMA = {
    "type": "record",
    "doc": "Classification of ATLAS stamps",
    "name": "atlas_stamp_classification_schema",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "classifications", "type": CLASSIFICATIONS},
        {"name": "classifier_name", "type": "string"},
        {"name": "classifier_version", "type": "string"},
        {
            "name": "brokerPublishTimestamp",
            "type": ["null", {"type": "long", "logicalType": "timestamp-millis"}],
            "doc": "timestamp of broker ingestion of ATLAS alert",
        },
        # {"name": "candid", "type": ["long", "string"]},
        # {"name": "mjd", "type": "double"},
        # {"name": "ra", "type": "double"},
        # {"name": "dec", "type": "double"},
        # {"name": "red", "type": ["null", "bytes"], "doc": "science stamp np.ndarray"},
        # {
        #     "name": "diff",
        #     "type": ["null", "bytes"],
        #     "doc": "difference stamp np.ndarray",
        # },
        # {"name": "FILTER", "type": "string"},
        # {"name": "AIRMASS", "type": "double"},
        # {"name": "SEEING", "type": "double"},
        # {"name": "SUNELONG", "type": "double"},
        # {"name": "MANGLE", "type": "double"},
    ],
}

## Data inside payload is a stringified dictionary with the following fields
# {
#     "collection": "The collection that will be written on",
#     "type": "insert" | "update",
#     "criteria": "JSON or dictionary which represent the filter of the query",
#     "data": "JSON or dictionary which represent the data to be inserted or updated"
# }

SCRIBE_SCHEMA = {
    "namespace": "db_operation",
    "type": "record",
    "name": "Command",
    "fields": [
        {"name": "payload", "type": "string"},
    ],
}
