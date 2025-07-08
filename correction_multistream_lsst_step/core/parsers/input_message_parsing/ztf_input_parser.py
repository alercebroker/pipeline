from typing import List, Dict
from .input_message_parsing import InputMessageParsingStrategy
import pandas as pd 
from core.schemas import schema_applier


#! TO BE MODIFIED TO FOLLOW LSST INPUT PARSER CLOSELY!!!!!!
#! PENDING BUT NOT THE PRIORITY RN!!!! :O
#! FJKSAJHDKJSHADJKHSAJKDHSA WE STILL TRYING TO KEEP THE STRUCTURE AS EASY TO HANDLE ZTF AS POSSIBLE


class ZTFInputMessageParser(InputMessageParsingStrategy):
    """
    Parser for ZTF (Zwicky Transient Facility) survey input messages.
    
    Key ZTF-specific characteristics:
    - Uses "detections" instead of "sources"
    - Uses "forced_photometries" instead of "forced_sources"  
    - Does not separate current from previous detections explicitly
    - No survey-specific additional objects (no ss_object/dia_object equivalents)
    - Uses different field names (e.g., "objectId" instead of "oid", "candid" instead of "measurement_id")
    
    This parser will normalize ZTF's structure into the same standard format
    used by LSST and other surveys in the pipeline.
    """
    
    def parse_input_messages(self, messages: List[dict]) -> Dict[str, any]:
        """
        Parse ZTF-specific input messages into standardized format.
        
        Args:
            messages (List[dict]): Raw ZTF messages. Each message typically contains:
                - objectId: ZTF object identifier (equivalent to LSST's "oid")
                - candid: Candidate ID (equivalent to LSST's "measurement_id")
                - detections: List of all detections for this object
                - forced_photometries: List of forced photometry measurements  
                - non_detections: List of upper limit measurements
                
        Returns:
            Dict[str, any]: Parsed data in standardized format. Note that:
                - All ZTF detections go into 'sources' (no previous_sources separation)
                - 'previous_sources' will be empty list for ZTF
                - 'additional_objects' will be empty dict (ZTF has no special objects)
        
        TODO: 
            This is a placeholder implementation. The actual ZTF message structure

        """
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
        """
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
    

#! TODO DO THE SCHEMAS FOR ZTF CORRECTLY AND IMPORT THEM 
def _apply_schema_or_empty(self, data: List[dict], schema: Dict) -> pd.DataFrame:
        """Apply schema to data or return empty DataFrame with correct columns."""
        if data:
            return schema_applier.apply_schema(data, schema)
        else:
            return pd.DataFrame(columns=list(schema.keys()))