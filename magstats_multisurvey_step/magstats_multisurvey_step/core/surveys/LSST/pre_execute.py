from typing import List, Dict, Tuple
import numpy as np
# Combine previous sources with sources
# Change name of coordinates errors to a standard ra_error and dec_error for all surveys
# LSST raErr and decErr are in degrees, while ra_error and dec_error are expected in arcsec
# (BaseObjectStatistics converts them to degrees and the resulting sigmara/sigmadec back to arcsec)
# Add a default value to ra_error and dec_error for LSST of DEFAULT_ERR_DEG to replace when raErr and decErr are null
# Add ss detections to the data extraction for LSST apart from detectons

DEG_TO_ARCSEC = 3600.0
# Fallback error (in degrees) for detections without raErr/decErr.
# Keeps the value previously used (0.01, which was effectively interpreted as degrees).
DEFAULT_ERR_DEG = 0.01


def extract_lsst_data(msg: Dict) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Extract LSST format with support for solar system detections.
    
    Returns:
        Tuple of (detections, forced_sources, non_detections, ss_detections)
    """
    
    def add_default_errors(source, default_value=DEFAULT_ERR_DEG):
        """Add ra_error and dec_error in arcsec, from raErr and decErr in degrees.

        Missing errors are replaced by `default_value` (in degrees).
        """
        ra_err = source.get("raErr")
        ra_err = ra_err if ra_err is not None else default_value
        source["ra_error"] = ra_err * DEG_TO_ARCSEC

        dec_err = source.get("decErr")
        dec_err = dec_err if dec_err is not None else default_value
        source["dec_error"] = dec_err * DEG_TO_ARCSEC
    
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