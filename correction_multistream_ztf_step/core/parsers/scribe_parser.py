

def scribe_parser(correction_list):

    result_messages = []
    for correction in correction_list:
        result_messages.append(
            {
                "step": "correction",
                "survey": "ztf",
                "payload": {
                    "oid": correction["oid"],
                    "detections": correction["detections"]
                }
            }
        )

    return result_messages




