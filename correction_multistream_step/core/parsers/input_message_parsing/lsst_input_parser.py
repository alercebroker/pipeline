from typing import List, Dict
from .input_message_parsing import InputMessageParsingStrategy
import pandas as pd
from core.schemas import schema_applier
import logging


class LSSTInputMessageParser(InputMessageParsingStrategy):
    """
    Parser for LSST survey input messages.
    
    Key LSST-specific characteristics:
    - Separates current sources from previous_sources explicitly
    - Includes forced_sources for photometry at known positions  
    - Provides non_detections for upper limit constraints
    - Contains survey-specific objects: ss_object (solar system) and dia_object (difference imaging)
    - Uses specific field names like "measurement_id", "oid", etc.
    
    The parser normalizes LSST's structure into the standard format expected
    by the rest of the correction pipeline, applying schemas to preserve precision.
    """
    
    def parse_input_messages(self, messages: List[dict]) -> Dict[str, any]:
        """
        Parse LSST-specific input messages into standardized pandas DataFrames with proper schemas.
        
        This method combines message parsing with schema application to ensure data precision
        is preserved. It returns ready-to-use pandas DataFrames instead of raw dictionaries.
        
        Args:
            messages (List[dict]): Raw LSST messages with complete structure
        
        Returns:
            Dict[str, any]: Parsed data with pandas DataFrames:
                - 'msg_data': DataFrame with basic message metadata
                - 'sources_df': DataFrame with current detections  
                - 'previous_sources_df': DataFrame with historical detections
                - 'forced_sources_df': DataFrame with forced photometry
                - 'non_detections_df': DataFrame with ndets
                - 'ss_objects_df': DataFrame with solar system objects
                - 'dia_objects_df': DataFrame with DIA objects
                - 'oids': Set of unique object IDs
                - 'measurement_ids': Dict mapping oids to measurement_id lists
        """
        logger = logging.getLogger(f"alerce.{self.__class__.__name__}")

        # Get raw parsed data first
        raw_data = self._parse_raw_messages(messages)
        
        # Get schemas
        schemas = self.get_input_schema_config()
        
        # Apply schemas to create proper DataFrames
        msg_df = pd.DataFrame(raw_data['msg_data'])
        
        # Apply schemas to each data type, handling empty cases
        sources_df = self._apply_schema_or_empty(
            raw_data['sources'], 
            schemas['sources_schema']
        )
        
        previous_sources_df = self._apply_schema_or_empty(
            raw_data['previous_sources'], 
            schemas['previous_sources_schema']
        )
        
        forced_sources_df = self._apply_schema_or_empty(
            raw_data['forced_sources'], 
            schemas['forced_sources_schema']
        )
        
        non_detections_df = self._apply_schema_or_empty(
            raw_data['non_detections'], 
            schemas['non_detections_schema']
        )
        
        ss_objects_df = self._apply_schema_or_empty(
            raw_data['ss_objects'], 
            schemas['ss_objects']
        )
        
        dia_objects_df = self._apply_schema_or_empty(
            raw_data['dia_objects'], 
            schemas['dia_objects']
        )
        
        # Get unique OIDs and measurement IDs for database queries
        oids = set(msg_df["oid"].unique())
        measurement_ids = (msg_df.groupby("oid")["measurement_id"]
                          .apply(lambda x: [str(id) for id in x]).to_dict())
        
        log_output = {
                'counts': {
                    'Current Sources': len(sources_df),
                    'Previous Sources': len(previous_sources_df),
                    'Forced Sources': len(forced_sources_df),
                    'Non-Detections': len(non_detections_df),
                    'SS Objects': len(ss_objects_df),
                    'DIA Objects': len(dia_objects_df)}
                }

        for data_type, count in log_output['counts'].items():
            logger.info(f"Received {count} {data_type}")

        parsed_input = {
            'data': {
                'msg_data': msg_df,
                'sources_df': sources_df,
                'previous_sources_df': previous_sources_df,
                'forced_sources_df': forced_sources_df,
                'non_detections_df': non_detections_df,
                'ss_objects_df': ss_objects_df,
                'dia_objects_df': dia_objects_df,
            },
            'oids': list(oids),
            'measurement_ids': measurement_ids
            }

        return parsed_input
    
    def _parse_raw_messages(self, messages: List[dict]) -> Dict[str, any]:
        """Extract raw data from messages without schema application."""
        # Initialize collectors for different data types
        all_sources = []           
        all_previous_sources = []  
        all_forced_sources = []    
        all_non_detections = []   
        all_ss_objects = []        
        all_dia_objects = []       
        msg_data = []              
        
        for msg in messages:    
            # Extract basic message identifiers
            oid = msg["oid"]                    
            measurement_id = msg["measurement_id"] 
            msg_data.append({"oid": oid, "measurement_id": measurement_id})
            
            # Parse previous sources
            for prev_source in msg["previous_sources"]:
                parsed_prv_source = {"new": True, **prev_source}
                all_previous_sources.append(parsed_prv_source)

            # Parse forced sources
            for f_source in msg["forced_sources"]:
                parsed_forced_source = {"new": True, **f_source}
                all_forced_sources.append(parsed_forced_source)

            # Parse non-detections
            for non_detection in msg["non_detections"]:
                parsed_non_detection = {**non_detection}
                all_non_detections.append(parsed_non_detection)

            # Parse ss objects
            ss_object = msg["ss_object"]
            if ss_object is not None:
                all_ss_objects.append({**ss_object})

            # Parse dia object
            dia_object = msg["dia_object"]
            if dia_object is not None:
                all_dia_objects.append({**dia_object})
            
            # Parse main source
            source = msg["source"]
            parsed_source = {"new": True, **source}
            all_sources.append(parsed_source)
        
        return {
            'msg_data': msg_data,
            'sources': all_sources,
            'previous_sources': all_previous_sources,
            'forced_sources': all_forced_sources,
            'non_detections': all_non_detections,
            'ss_objects': all_ss_objects,
            'dia_objects': all_dia_objects
            
        }
    
    def _apply_schema_or_empty(self, data: List[dict], schema: Dict) -> pd.DataFrame:
        """Apply schema to data or return empty DataFrame with correct columns."""
        if data:
            return schema_applier.apply_schema(data, schema)
        else:
            return pd.DataFrame(columns=list(schema.keys()))
    
    def get_input_schema_config(self) -> Dict[str, any]:
        """Return LSST-specific pandas schema configuration for precise data handling."""
        from core.schemas.LSST.LSST_schemas import (
            dia_non_detection_limit_schema,   
            dia_forced_source_schema,         
            dia_source_schema,                
            ss_object_schema,                 
            dia_object_schema                
        )
        
        return {
            'sources_schema': dia_source_schema,
            'previous_sources_schema': dia_source_schema,
            'forced_sources_schema': dia_forced_source_schema,
            'non_detections_schema': dia_non_detection_limit_schema,
            'ss_objects': ss_object_schema,     
            'dia_objects': dia_object_schema    
            
        }