from typing import List


def extract_lightcurve_from_messages(messages: List[dict]):
    """Extract detections and non detections from messages.

    Parameters
    ----------
    messages : List[dict]
        List of messages from the correction step.

    Returns
    -------
    oids : dict
        Dictionary with the oid as key and the detections and non detections

    Examples
    --------
    >>> messages = [
    ...     {
    ...         "oid": "ZTF18abtjpiw",
    ...         "detections": [
    ...             {
    ...                 "candid": 1115754325515015016,
    ...                 "fid": 1,
    ...                 "oid": "ZTF18abtjpiw",
    ...                 ...
    ...             },
    ...         ],
    ...         "non_detections": [
    ...             {
    ...                 "fid": 1,
    ...                 "oid": "ZTF18abtjpiw",
    ...                 "mjd": 58350.0,
    ...                 "diffmaglim": 20.5,
    ...             }
    ...         ],
    ...     },
    ... ]
    >>> oids = extract_detections_from_messages(messages)
    >>> oids["ZTF18abtjpiw"]["detections"][0]["candid"]
    1115754325515015016
    >>> oids["ZTF18abtjpiw"]["non_detections"][0]["diffmaglim"]
    20.5
    """
    oids = {}
    for msg in messages:
        dets = msg["detections"]
        non_dets = msg["non_detections"]
        oids[msg["oid"]] = {
            "detections": dets,
            "non_detections": non_dets,
        }

    return oids


def get_candids_from_messages(messages: List[dict]):
    candids = {}
    for msg in messages:
        if msg["oid"] not in candids:
            candids[msg["oid"]] = []
        candids[msg["oid"]].extend(msg["candid"])
    return candids
