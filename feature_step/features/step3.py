import pandas as pd
import logging
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from apf.core import get_class
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer
from .database import (
    PSQLConnection,
    get_sql_references,
)


class FeatureStep(GenericStep):
    """FeatureStep3 Description
    
    A third step that consumes messages in pre_execute method and calculates the number of detections
    by counting sources and previous sources.

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(
        self,
        config=None,
        db_sql: PSQLConnection = None,
        **step_args,
    ):
        super().__init__(config=config, **step_args)
        
        self.logger = logging.getLogger("alerce.FeatureStep3")
        self.survey = self.config.get("SURVEY", "ztf")
        
        # Create detections_staging folder if it doesn't exist
        self.detections_folder = os.path.join(os.path.dirname(__file__), '..', 'detections_staging')
        os.makedirs(self.detections_folder, exist_ok=True)
        self.logger.info(f"Detection results will be saved to: {self.detections_folder}")

    def calculate_detections(self, message):
        """Calculate number of detections from sources and previous sources"""
        try:
            detection_data = {
                'sources_count': 0,
                'previous_sources_count': 0,
                'total_detections': 0
            }
            
            # Count sources directly from message
            sources = message.get('sources', [])
            sources_count = len(sources) if sources else 0
            detection_data['sources_count'] = sources_count
            
            # Count previous sources directly from message
            previous_sources = message.get('previous_sources', [])
            previous_sources_count = len(previous_sources) if previous_sources else 0
            detection_data['previous_sources_count'] = previous_sources_count
            
            # Calculate total detections
            total_detections = sources_count + previous_sources_count
            detection_data['total_detections'] = total_detections
            
            return detection_data
                
        except Exception as e:
            self.logger.error(f"Error calculating detections: {e}")
        
        return detection_data

    def pre_execute(self, messages: List[dict]):
        """Pre-execute method that calculates detections from the list of messages"""
        self.logger.info(f"Starting pre_execute - calculating detections for {len(messages)} messages")
        
        try:
            # Calculate detections for each message
            detection_results = []
            for i, message in enumerate(messages):
                detection_data = self.calculate_detections(message)
                detection_results.append(detection_data)
                
                self.logger.info(f"Message {i+1}: Sources={detection_data['sources_count']}, "
                               f"Previous Sources={detection_data['previous_sources_count']}, "
                               f"Total={detection_data['total_detections']}")
            
            # Save detection counts
            self.save_detection_counts(detection_results)
            
        except Exception as e:
            self.logger.error(f"Error calculating detections in pre_execute: {e}")
        
        # Return the original messages unchanged
        return messages

    def save_detection_counts(self, detection_results):
        """Save detection counts - minimal info per message"""
        try:
            import hashlib
            import random
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Add random hash to ensure unique filenames
            random_hash = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
            
            # Save minimal data - just the detection counts per message
            data = {
                'timestamp': datetime.now().isoformat(),
                'total_messages': len(detection_results),
                'messages': detection_results
            }
            
            # Save to file with timestamp and random hash
            filename = f"detection_counts_{timestamp}_{random_hash}.json"
            filepath = os.path.join(self.detections_folder, filename)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Detection counts saved to: {filepath}")
            self.logger.info(f"Processed {len(detection_results)} messages")
            
        except Exception as e:
            self.logger.error(f"Error saving detection counts: {e}")

    def save_detection_results(self, detection_data, oid=None):
        """Save detection results to detections_staging folder - DEPRECATED"""
        # This method is no longer used but kept for compatibility
        pass

    def save_batch_summary(self, batch_results):
        """Save a summary of the batch detection results - DEPRECATED"""
        # This method is no longer used but kept for compatibility
        pass

    def execute(self, messages):
        """Execute method - placeholder for now"""
        self.logger.info(f"Executing with {len(messages)} messages")
        
        # For now, just return the messages as-is
        output = []
        for message in messages:
            output.append({
                "oid": message.get("oid", "unknown"),
                "processed": True,
                "survey": self.survey,
                "step": "detection_calculation"
            })
        
        return output

    def post_execute(self, result):
        """Post-execute method"""
        self.logger.info(f"Post-processing {len(result)} results")
        return result

    def tear_down(self):
        """Cleanup method"""
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        
        if hasattr(self, 'producer'):
            self.producer.__del__()