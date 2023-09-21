# to make this work, we'll have to take the original Mongo Commands, assume they're Ok
# and move on with that 

class ValidCommands:
    insert_object = "insert_object" # insert + object
    insert_detections = "insert_detections" # update + detections
    upsert_probabilities = "upsert_probabilities" # update + probabilities
    upsert_features = "update_features" # update + features
    upsert_non_detections = "upsert_non_detections" # update + non_detections


