from typing import List, Dict
from .input_message_parsing import InputMessageParsingStrategy
import pandas as pd 
from core.schemas import schema_applier
import logging


class ZTFInputMessageParser(InputMessageParsingStrategy):
    """
    Parser for ZTF (Zwicky Transient Facility) survey input messages.
    
    Key ZTF-specific characteristics:
    - Uses "detections" instead of "sources"
    - Uses "forced_photometries" instead of "forced_sources"  
    - No survey-specific additional objects (no ss_object/dia_object equivalents)

    This parser will normalize ZTF's structure into the same standard format
    used by LSST and other surveys in the pipeline.
    """
    
    def parse_input_messages(self, messages: List[dict]) -> Dict[str, any]:
        """
        Parse ZTF-specific input messages into standardized format.
        
        Args:
            messages (List[dict]): Raw ZTF messages. Each message typically contains:
                - oid: ZTF object identifier 
                - measurement_id: ZTF measurement identifier
                - detections: List of all detections for this object
                - forced_photometries: List of forced photometry measurements  
                - non_detections: List of non-detections
                
        Returns:
            Dict[str, any]: Parsed data in standardized format. Note that:
                - All ZTF detections go into 'sources' (no previous_sources separation)
                - 'additional_objects' will be empty dict (ZTF has no special objects like LSST dia object or dia object)
        
        """
        # Initialize collectors - simpler structure than LSST
        logger = logging.getLogger(f"alerce.{self.__class__.__name__}")

        # Get raw parsed data first
        raw_data = self._parse_raw_messages(messages)
        
        # Get schemas
        schemas = self.get_input_schema_config()
        
        # Apply schemas to create proper DataFrames
        msg_df = pd.DataFrame(raw_data['msg_data'])
        
        # Apply schemas to each data type, handling empty cases
        sources_df = self._apply_schema_or_empty(
            raw_data['detections'], 
            schemas['detections_schema']
        )
        
        previous_detections_df = self._apply_schema_or_empty(
            raw_data['previous_detections'], 
            schemas['previous_detections_schema']
        )
        
        forced_photometries_df = self._apply_schema_or_empty(
            raw_data['forced_photometries'], 
            schemas['forced_photometries_schema']
        )
        
        non_detections_df = self._apply_schema_or_empty(
            raw_data['non_detections'], 
            schemas['non_detections_schema']
        )
        
        # Get unique OIDs and measurement IDs for database queries
        oids = set(msg_df["oid"].unique())
        measurement_ids = (msg_df.groupby("oid")["measurement_id"]
                          .apply(lambda x: [str(id) for id in x]).to_dict())
        
        log_output = {
                'counts': {
                    'Current detections': len(sources_df),
                    'Previous detections': len(previous_detections_df),
                    'Forced photometries': len(forced_photometries_df),
                    'Non-Detections': len(non_detections_df)}
                }

        for data_type, count in log_output['counts'].items():
            logger.info(f"Received {count} {data_type}")

        parsed_input = {
            'data': {
                'msg_data': msg_df,
                'detections_df': sources_df,
                'previous_detections_df': previous_detections_df,
                'forced_photometries_df': forced_photometries_df,
                'non_detections_df': non_detections_df
            },
            'oids': list(oids),
            'measurement_ids': measurement_ids
            }

        return parsed_input
    def _parse_raw_messages(self, messages: List[dict]) -> Dict[str, any]:
        """Extract raw data from messages without schema application."""
        # Initialize collectors for different data types
        all_detections = []           
        all_previous_detections = []  
        all_forced_photometries = []    
        all_non_detections = []        
        msg_data = []              
        for msg in messages:    
            # Extract basic message identifiers
            oid = msg["oid"]                    
            measurement_id = msg["measurement_id"] 
            msg_data.append({"oid": oid, "measurement_id": measurement_id})
            
            # Parse previous sources
            for prev_det in msg["prv_detections"]:
                parsed_prv_source = {"new": True, **prev_det}
                all_previous_detections.append(parsed_prv_source)

            # Parse forced sources
            for fphot in msg["forced_photometries"]:
                parsed_forced_photometry = {"new": True, **fphot}
                all_forced_photometries.append(parsed_forced_photometry)

            # Parse non-detections
            for non_detection in msg["non_detections"]:
                parsed_non_detection = {**non_detection}
                all_non_detections.append(parsed_non_detection)

            # Parse main source
            for detection in msg["detections"]:
                parsed_detection = {"new": True, **detection}
                all_detections.append(parsed_detection)

        return {
            'msg_data': msg_data,
            'detections': all_detections,
            'previous_detections': all_previous_detections,
            'forced_photometries': all_forced_photometries,
            'non_detections': all_non_detections
        }
    
    def get_input_schema_config(self) -> Dict[str, any]:
        """Return ZTF-specific pandas schema configuration for precise data handling."""
        from core.schemas.ZTF.ZTF_schemas import (
            non_detection_schema,   
            candidate_schema,         
            prv_candidate_schema,                
            forced_photometry_schema         
        )
        
        return {
            'detections_schema': candidate_schema,
            'previous_detections_schema': prv_candidate_schema,
            'forced_photometries_schema': forced_photometry_schema,
            'non_detections_schema': non_detection_schema        
        }
    

    def _apply_schema_or_empty(self, data: List[dict], schema: Dict) -> pd.DataFrame:
        """Apply schema to data or return empty DataFrame with correct columns."""
        if data:
            return schema_applier.apply_schema(data, schema)
        else:
            return pd.DataFrame(columns=list(schema.keys()))
        
