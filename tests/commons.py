PIPELINE_ORDER = {
    "ATLAS": {"S3Step": None, "SortingHatStep": {"IngestionStep": None}},
    "ZTF": {
        "EarlyClassifier": None,
        "S3Step": None,
        "WatchlistStep": None,
        "SortingHatStep": {
            "IngestionStep": {
                "XmatchStep": {"FeaturesComputer": {"LateClassifier": None}}
            }
        },
    },
}
