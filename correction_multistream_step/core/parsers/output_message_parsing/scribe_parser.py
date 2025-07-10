
def scribe_parser_survey(correction_list, survey):
    """
    Function to parse the corrections for the different surveys
    Input:
        correction_list: list of corrections
        survey: name of the survey
    Output:
        result_messages: list of messages parsing the corrections into the correct format for the scribe
    """
    result_messages = []
    for correction in correction_list:
        if survey == "lsst":
            result_messages.append(
                {
                    "step": "correction",
                    "survey": survey,
                    "payload": {
                        "oid": correction["oid"],
                        "measurement_id": correction["measurement_id"],
                        "sources": correction["sources"],
                        "previous_sources": correction["previous_sources"],
                        "forced_sources": correction["forced_sources"],
                        "non_detections": correction["non_detections"],
                        "dia_object": correction["dia_object"],
                        "ss_object": correction["ss_object"]
                    }
                }
            )
        if survey == "ztf":
            result_messages.append(
                {
                    "step": "correction",
                    "survey": survey,
                    "payload": {
                        "oid": correction["oid"],
                        "measurement_id": correction["measurement_id"],
                        "candidates": correction["candidates"],
                        "previous_candidates": correction["previous_candidates"],
                        "forced_photometries": correction["forced_photometries"],
                        "non_detections": correction["non_detections"]
                    }
                }
            )
    return result_messages




