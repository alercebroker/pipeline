from typing import List, Dict, Tuple
import numpy as np
# Combine previous sources with sources
# Change name of coordinates errors to a standard ra_error and dec_error for all surveys 
# Add a default value to ra_error and dec_error for LSST of 0.01 to replace when raErr and decErr are null
# Add ss detections to the data extraction for LSST apart from detectons

def extract_lsst_data(msg: Dict) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Extract LSST format with support for solar system detections.
    
    Returns:
        Tuple of (detections, forced_sources, non_detections, ss_detections)
    """
    
    def add_default_errors(source, default_value=0.01):
        """Add error fields with default values if missing."""
        ra_err = source.get("raErr")
        source["ra_error"] = ra_err if ra_err is not None else default_value
        
        dec_err = source.get("decErr")
        source["dec_error"] = dec_err if dec_err is not None else default_value
    
    # Process regular sources
    sources = msg.get("sources", [])
    for source in sources:
        add_default_errors(source)
    
    # Process and extend with previous sources
    prev_sources = msg.get("previous_sources", [])
    for source in prev_sources:
        add_default_errors(source)
    sources.extend(prev_sources)
    
    # Process solar system detections (new)
    ss_detections = msg.get("ss_sources", [])
    for ss_det in ss_detections:
        add_default_errors(ss_det)
    
    # Process forced sources
    forced_sources = msg.get("forced_sources", [])
    
    # Non-detections
    non_detections = msg.get("non_detections", [])
    
    return sources, forced_sources, non_detections, ss_detections