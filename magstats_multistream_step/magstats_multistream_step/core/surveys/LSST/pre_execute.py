from typing import List, Dict, Tuple

# Combine previous sources with sources
# Change name of coordinates errors to a standard ra_error and dec_error for all surveys 
# Add a default value to ra_error and dec_error for LSST of 0.01 to replace when raErr and decErr are null

def extract_lsst_data(msg: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Extract LSST format: source + previous_sources + forced_sources = Detections (used in the statistics calculator)
    Add ra_error as a field from raErr and decErr in the original fields names"""    
    
    def add_default_errors(source, default_value=0.01):
        """Add error fields with default values if missing."""
        source["ra_error"] = source.get("raErr") or default_value
        source["dec_error"] = source.get("decErr") or default_value
    
    # Process sources
    sources = msg.get("sources", [])
    for source in sources:
        add_default_errors(source)

    # Process and extend with previous sources
    prev_sources = msg.get("previous_sources", [])
    for source in prev_sources:
        add_default_errors(source)
    
    sources.extend(prev_sources)
    
    # Process forced sources
    forced_sources = msg.get("forced_sources", [])
    
    non_detections = msg.get("non_detections", [])
    return sources, forced_sources, non_detections
