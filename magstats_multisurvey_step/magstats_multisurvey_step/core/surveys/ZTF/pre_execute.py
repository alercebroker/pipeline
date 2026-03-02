import copy
from typing import Dict, List, Tuple


def extract_ztf_data(msg: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Extract ZTF format: detections + forced_photometries + non_detections

    - Adds mjdendref from extra fields back into main fields
    - Standardizes coordinate error field names to ra_error and dec_error
    - Separates detections into forced and non-forced

    Returns:
        Tuple of (detections, forced_photometries, non_detections)
    """
    detections = []
    forced_photometries = []

    for det in msg.get("detections", []):
        det_copy = copy.deepcopy(det)

        # calculate mean ra in base objectstats is set to use raErr and decErr format (a selector can be created to replace these two lines)
        det_copy["ra_error"] = det_copy["e_ra"]
        det_copy["dec_error"] = det_copy["e_dec"]
        # convert mjdendref to MJD format
        jdendref = det_copy.get("jdendref")
        if jdendref:
            det_copy["mjdendref"] = jdendref - 2400000.5

        # Separate forced to its own list
        if det_copy.get("forced"):
            forced_photometries.append(det_copy)
        else:
            detections.append(det_copy)

    for det in msg.get("previous_detections", []):
        det_copy = copy.deepcopy(det)
        # calculate mean ra in base objectstats is set to use raErr and decErr format (a selector can be created to replace these two lines)
        det_copy["ra_error"] = det_copy["e_ra"]
        det_copy["dec_error"] = det_copy["e_dec"]
        # convert mjdendref to MJD format
        jdendref = det_copy.get("jdendref")
        if jdendref:
            det_copy["mjdendref"] = jdendref - 2400000.5
        detections.append(det_copy)

    for det in msg.get("forced_photometries", []):
        det_copy = copy.deepcopy(det)
        # calculate mean ra in base objectstats is set to use raErr and decErr format (a selector can be created to replace these two lines)
        det_copy["ra_error"] = det_copy["e_ra"]
        det_copy["dec_error"] = det_copy["e_dec"]
        forced_photometries.append(det_copy)

    non_detections = msg.get("non_detections", [])

    return detections, forced_photometries, non_detections
