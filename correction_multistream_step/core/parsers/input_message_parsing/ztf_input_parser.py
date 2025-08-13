from typing import List, Dict
from .input_message_parsing import InputMessageParsingStrategy
import pandas as pd 
from core.schemas import schema_applier


#! TO BE MODIFIED TO FOLLOW LSST INPUT PARSER CLOSELY!!!


class ZTFInputMessageParser(InputMessageParsingStrategy):
    """
    """
    
    def parse_input_messages(self, messages: List[dict]) -> Dict[str, any]:
        """
    
        TODO: 
            This is a placeholder implementation. The actual ZTF message structure

        
        # Initialize collectors - simpler structure than LSST
        all_detections = []        # All ZTF detections (current + historical)
        all_non_detections = []    # Upper limits when objects weren't detected
        all_forced_photometries = [] # Forced photometry at known positions
        msg_data = []              # Basic message metadata
        
        for msg in messages:
        
            pass
        
        return {
        'data': {
            'detections_df': detections_df,
            'non_detections_df': non_detections_df,
            'forced_photometry_df': forced_photometry_df,
        },
        'counts': {
            'Detections': len(detections_df),
            'Non-Detections': len(non_detections_df),
            'Forced Photometry': len(forced_photometry_df)
        },
        'oids': list(oids),
        'measurement_ids': measurement_ids
    }
    
    def get_input_schema_config(self) -> Dict[str, any]:
        
        Return ZTF-specific pandas schema configuration for precise data handling.
                
        Returns:
            Dict[str, any]: ZTF schemas
        
        TODO:
            This is a placeholder. The actual implementation needs:

        """
        # TODO: Import ZTF schemas when they are created

        return {
            # ZTF detections schema - all detections use same structure
            'sources_schema': {},  # TODO: Replace with ztf_detection_schema
            
            # ZTF doesn't separate previous sources, so this is unused
            'previous_sources_schema': {},  
            
            # ZTF forced photometry schema
            'forced_sources_schema': {},  # TODO: Replace with ztf_forced_photometry_schema
            
            # ZTF non-detection/upper limit schema
            'non_detections_schema': {},  # TODO: Replace with ztf_non_detection_schema
            
            # ZTF has no additional survey-specific objects
        }
    

#TODO
def _apply_schema_or_empty(self, data: List[dict], schema: Dict) -> pd.DataFrame:
        """Apply schema to data or return empty DataFrame with correct columns."""
        if data:
            return schema_applier.apply_schema(data, schema)
        else:
            return pd.DataFrame(columns=list(schema.keys()))