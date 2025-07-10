from typing import List, Dict, Tuple

def extract_lsst_data(msg: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Extract LSST format: source + previous_sources + forced_sources = Detections (used in the statistics calculator)
    Add ra_error as a field from raErr and decErr in the original fields names"""    
    
    # Combine previous sources with sourcees
    sources = msg.get("sources", [])
    for source in sources:
        source["ra_error"] = source["raErr"]
        source["dec_error"] = source["decErr"]
    
    # Process previous sources and add to main sources
    prev_sources = msg.get("previous_sources", [])
    for source in prev_sources:
        source["ra_error"] = source["raErr"]
        source["dec_error"] = source["decErr"]
    sources.extend(prev_sources)
    
    # Process forced sources
    forced_sources = msg.get("forced_sources", [])
    
    non_detections = msg.get("non_detections", [])
    return sources, forced_sources, non_detections
