# to make this work, we'll have to take the original Mongo Commands, assume they're Ok
# and move on with that


class ValidCommands:
    # insert + object
    insert_object = "insert_object"
    # update + object no xmatch in data
    update_object_from_stats = "update_object_from_stats"
    # update + detections
    insert_detections = "insert_detections"
    # update_probabilities + probabilities in data
    upsert_probabilities = "upsert_probabilities"
    # update_features + features in data
    upsert_features = "upsert_features"
    # update + non_detections
    upsert_non_detections = "upsert_non_detections"
    # upsert + magstats PSQL EXCLUSIVE
    update_object_stats = "update_object_stats"
    # update + object + xmatch in data
    upsert_xmatch = "upsert_xmatch"
    # update + forced_photometry
    insert_forced_photo = "insert_forced_photo"
    # upsert + score
    upsert_score = "upsert_score"
