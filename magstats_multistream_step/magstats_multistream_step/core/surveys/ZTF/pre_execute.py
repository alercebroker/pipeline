import copy
from typing import List, Dict, Tuple

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
        
        # Standardize coordinate error field names
        det_copy["ra_error"] = det_copy.get("e_ra")
        det_copy["dec_error"] = det_copy.get("e_dec")
        
        # Extract mjdendref from extra_fields if present (only in alert messages)
        jdendref = det_copy.get("extra_fields", {}).get("jdendref")
        if jdendref:
            det_copy["mjdendref"] = jdendref - 2400000.5
        
        # Separate forced to its own list
        if det_copy.get("forced"):
            forced_photometries.append(det_copy)
        else:
            detections.append(det_copy)
    
    non_detections = msg.get("non_detections", [])
    
    return detections, forced_photometries, non_detections
