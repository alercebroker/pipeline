CLASSIFICATIONS = {
    "name": "classifications",
    "type": {
        "type": "array",
        "items": {
            "type": "record",
            "name": "classificationDict",
            "fields": [
                {"name": "classifier_name", "type": "string"},
                {"name": "model_version", "type": "string"},
                {"name": "class_name", "type": "double"},
                {"name": "probability",  "type": "double"}
            ]
        }
    }
}

SCHEMA = {
    "type": "record",
    "doc": "Classification of ATLAS stamps",
    "name": "atlas_stamp_classification_schema",
    "fields": [
        {"name": "oid", "type": "string"},
        {"name": "classifications", "type": CLASSIFICATIONS},
        {"name": "model_version", "type": "string"},
        {"name": "brokerPublishTimestamp",
         "type": ["null", {"type": "long", "logicalType": "timestamp-millis"}],
         "doc": "timestamp of broker ingestion of ATLAS alert"},
        {"name": "candid", "type": ["long", "string"]},
        {"name": "mjd", "type": "double"},
        {"name": "ra", "type": "double"},
        {"name": "dec", "type": "double"},
        {"name": "red",
         "type": ["null", "bytes"],
         "doc": "science stamp np.ndarray"},
        {"name": "diff",
         "type": ["null", "bytes"],
         "doc": "difference stamp np.ndarray"},
        {"name": "FILTER", "type": "string"},
        {"name": "AIRMASS", "type": "double"},
        {"name": "SEEING", "type": "double"},
        {"name": "SUNELONG", "type": "double"},
        {"name": "MANGLE", "type": "double"},
    ],
}
