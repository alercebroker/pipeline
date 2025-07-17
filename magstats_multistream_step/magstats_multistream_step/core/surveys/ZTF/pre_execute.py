import copy
from typing import List, Dict, Tuple

def extract_ztf_data(msg: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Extract ZTF format: detections + non_detections fields and adds mjdendref from extra fields back into main fields 
    (Temporal until ZTF sends all the data in the main fields separated)
    Change names for all detections error coordinates fields, to fit raErrand decErr format
    """
    detections = []
    forced_photometries = []
    for det in msg.get("detections", []):
        det_copy = copy.deepcopy(det)
        
        # calculate mean ra in base objectstats is set to use raErr and decErr format (a selector can be created to replace these two lines)
        det_copy["ra_error"] = det_copy["e_ra"]
        det_copy["dec_error"] = det_copy["e_dec"]
        
        jdendref = det_copy.get("extra_fields", {}).get("jdendref")
        if jdendref:
            det_copy["mjdendref"] = jdendref - 2400000.5
        if det_copy["forced"]:
            forced_photometries.append(det_copy)
        else:
            detections.append(det_copy)
    
    non_detections = msg.get("non_detections", [])
    return detections, forced_photometries, non_detections
