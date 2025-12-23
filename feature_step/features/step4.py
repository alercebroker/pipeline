import json
import logging
import os
from typing import Any, Dict, List
from pathlib import Path
import pandas as pd
import time
import hashlib
import random

from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer
from .database import (
    PSQLConnection)

class FeatureStep(GenericStep):
    """FeatureStep4 Description
    
    A fourth step that reads messages from simulated_jsons folders and returns them as output,
    adding the class name field extracted from the folder structure.

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
        
        self.logger = logging.getLogger("alerce.FeatureStep4")
        self.survey = self.config.get("SURVEY", "ztf")
        
        # Batch counter - increments with each execute call
        self.batch = 0
        
        # Get the simulated_parquet folder path (use SIMWv5 simulated output)
        self.simulated_parquet_folder = os.path.join(
            os.path.dirname(__file__), '..', 'simulated_parquet_SIMWv5'
        )
        self.logger.info(f"Simulated Parquet folder: {self.simulated_parquet_folder}")
        
        # Create output directory for saved messages
        self.output_messages_dir = os.path.join(
            os.path.dirname(__file__), '..', 'output_messages'
        )
        os.makedirs(self.output_messages_dir, exist_ok=True)
        self.logger.info(f"Output messages directory: {self.output_messages_dir}")
        
        # Generator and batch size for simulating consumption
        self.message_generator = None
        self.batch_size = self.config.get("BATCH_SIZE", 1)  # Default batch size: 1 message per batch
        self.total_consumed = 0

    def _get_class_name_from_folder(self, folder_name):
        """Extract class name from folder name by removing date suffix"""
        # Example: AGN_DDF_20251010 -> AGN
        class_name_parts = folder_name.split('_')
        if 'DDF' in class_name_parts:
            ddf_index = class_name_parts.index('DDF')
            return '_'.join(class_name_parts[:ddf_index])
        if 'WFD' in class_name_parts:
            wfd_index = class_name_parts.index('WFD')
            return '_'.join(class_name_parts[:wfd_index])
        return folder_name

    def message_stream_generator(self):
        """Generator that yields messages one by one from simulated_parquet files"""
        if not os.path.exists(self.simulated_parquet_folder):
            self.logger.warning(f"Simulated Parquet folder not found: {self.simulated_parquet_folder}")
            return
        
        # Iterate through all class folders
        class_folders = sorted(os.listdir(self.simulated_parquet_folder))
        
        for class_folder in class_folders:
            class_folder_path = os.path.join(self.simulated_parquet_folder, class_folder)
            
            # Skip if not a directory
            if not os.path.isdir(class_folder_path):
                continue
            
            # Extract class name
            class_name = self._get_class_name_from_folder(class_folder)
            self.logger.info(f"Streaming from class: {class_name} (folder: {class_folder})")
            
            # Read the parquet files
            metadata_file = os.path.join(class_folder_path, 'messages_metadata.parquet')
            sources_file = os.path.join(class_folder_path, 'sources.parquet')
            
            if not os.path.exists(metadata_file) or not os.path.exists(sources_file):
                self.logger.warning(f"Missing parquet files in {class_folder_path}")
                continue
            
            try:
                # Read parquet files
                metadata_df = pd.read_parquet(metadata_file)
                sources_df = pd.read_parquet(sources_file)
                
                self.logger.info(f"Loaded {len(metadata_df)} objects from {class_folder}")
                
                # Group sources by oid
                sources_by_oid = sources_df.groupby('oid')
                
                # Iterate through each object in metadata
                for idx, row in metadata_df.iterrows():
                    oid = int(row['oid'])
                    measurement_ids = row.get('measurement_ids', [])
                    
                    # Ensure measurement_ids is a list of ints
                    if isinstance(measurement_ids, (list, tuple)):
                        measurement_ids = [int(mid) for mid in measurement_ids]
                    else:
                        measurement_ids = []
                    
                    # Get sources for this oid
                    if oid in sources_by_oid.groups:
                        object_sources = sources_by_oid.get_group(oid)
                        
                        # Define fields allowed in source_2 schema
                        allowed_source_fields = {
                            'oid', 'sid', 'measurement_id', 'mjd', 'band',
                            'diaObjectId', 'ra', 'dec', 'psfFlux', 'psfFluxErr',
                            'scienceFlux', 'scienceFluxErr', 'x', 'y'
                        }
                        
                        # Convert to list of dicts, handling NaN values and filtering fields
                        sources_list = []
                        for _, source_row in object_sources.iterrows():
                            source_dict = {}
                            for k, v in source_row.to_dict().items():
                                if k not in allowed_source_fields:
                                    continue
                                
                                # Handle None/NaN values
                                if pd.isna(v):
                                    source_dict[k] = None
                                    continue
                                
                                # Convert types according to schema
                                if k == 'oid':
                                    source_dict[k] = int(v)
                                elif k == 'sid':
                                    source_dict[k] = int(v)
                                elif k == 'measurement_id':
                                    source_dict[k] = int(v)
                                elif k == 'band':
                                    source_dict[k] = int(v) if v is not None else None
                                elif k == 'diaObjectId':
                                    source_dict[k] = int(v) if v is not None else None
                                elif k in ['mjd', 'ra', 'dec', 'psfFlux', 'psfFluxErr', 
                                          'scienceFlux', 'scienceFluxErr', 'x', 'y']:
                                    source_dict[k] = float(v)
                                else:
                                    source_dict[k] = v
                            
                            sources_list.append(source_dict)
                    else:
                        sources_list = []
                    
                    # Extract period from metadata (convert sentinel / NaN to None)
                    period = row.get('period', None)
                    try:
                        # Handle pandas NaN
                        if pd.isna(period):
                            period = None
                    except Exception:
                        # pd.isna may fail for non-pandas types; ignore
                        pass
                    # Handle sentinel value used when period is missing
                    if period == -999.0:
                        period = None

                    # Build message structure according to output2.avsc schema
                    message = {
                        'oid': oid,  # long type
                        'measurement_id': measurement_ids,  # array of long
                        'sources': sources_list,
                        'previous_sources': [],
                        'forced_sources': [],
                        'dia_object': [],
                        'class_name': class_name,
                        'period': float(period) if period is not None else None,
                    }
                    
                    yield message
                
                self.logger.debug(f"Finished streaming from {class_folder}")
                    
            except Exception as e:
                self.logger.error(f"Error reading parquet files in {class_folder_path}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        self.logger.info("Finished streaming all messages")

    def get_next_batch(self):
        """Get the next batch of messages from the generator, simulating Kafka consumption"""
        # Initialize generator on first call
        if self.message_generator is None:
            self.logger.info("Initializing message stream generator")
            self.message_generator = self.message_stream_generator()
        
        # Collect batch_size messages from generator
        batch = []
        try:
            for _ in range(self.batch_size):
                message = next(self.message_generator)
                batch.append(message)
        except StopIteration:
            # Generator exhausted
            if len(batch) == 0:
                self.logger.info(f"All messages consumed. Total: {self.total_consumed}")
            pass
        
        self.total_consumed += len(batch)
        
        if len(batch) > 0:
            self.logger.info(
                f"Consumed batch of {len(batch)} messages "
                f"(total consumed so far: {self.total_consumed})"
            )
        
        return batch

    def _save_message_to_json(self, message):
        """Save message to JSON file with timestamp+randomhash as filename"""
        try:
            # Generate timestamp (milliseconds)
            timestamp = int(time.time() * 1000)
            
            # Generate random hash
            random_string = f"{random.random()}_{message.get('oid', 'unknown')}"
            hash_object = hashlib.md5(random_string.encode())
            random_hash = hash_object.hexdigest()[:8]  # Use first 8 chars
            
            # Create filename
            filename = f"{timestamp}_{random_hash}.json"
            filepath = os.path.join(self.output_messages_dir, filename)
            
            # Convert message to JSON string to get size
            json_string = json.dumps(message, indent=2, default=str)
            json_size_bytes = len(json_string.encode('utf-8'))
            json_size_kb = json_size_bytes / 1024
            
            # Save message to JSON
            with open(filepath, 'w') as f:
                f.write(json_string)
            
            self.logger.info(f"Saved message to {filename} - Size: {json_size_kb:.2f} KB ({json_size_bytes} bytes)")
            
        except Exception as e:
            self.logger.error(f"Error saving message to JSON: {e}")

    def pre_execute(self, messages: List[dict]):
        """Pre-execute method - allows continuous batch processing"""
        # Increment batch counter
        self.batch += 1
        
        self.logger.info(f"Pre-execute batch {self.batch}")
        return messages

    def execute(self, messages):
        """Execute method - returns next batch of simulated messages with class_name field"""        
        self.logger.info(f"Executing step4 - batch {self.batch} - getting next batch from queue")
        
        # Get next batch of simulated messages
        batch_messages = self.get_next_batch()
        
        # Return the batch as output
        return batch_messages

    def post_execute(self, result):
        """Post-execute method"""
        self.logger.info(f"Post-processing {len(result)} results")
        
        # Save each message to JSON (uncomment to enable)
        #for message in result:
        #     self._save_message_to_json(message)
        
        return result

    def tear_down(self):
        """Cleanup method"""
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        
        if hasattr(self, 'producer'):
            self.producer.__del__()
