import pandas as pd
import logging
import json
import os
import hashlib
import random
from typing import Any, Dict, Iterable, List, Optional

from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer
from .database import (
    PSQLConnection,
    get_sql_references,
)


class FeatureStep(GenericStep):
    """FeatureStep2 Description
    
    A step that processes incoming messages and extracts features from all of them.
    No secondary consumer is used - messages are processed directly in the execute method.

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
        
        self.logger = logging.getLogger("alerce.FeatureStep2")
        self.survey = self.config.get("SURVEY", "ztf")
        
        # Load feature name mapping
        self.feature_mapping = self._load_feature_mapping()
        
        # Create dfs_staging folder if it doesn't exist
        self.dfs_folder = os.path.join(os.path.dirname(__file__), '..', 'dfs_staging')
        os.makedirs(self.dfs_folder, exist_ok=True)
        self.logger.info(f"DataFrames will be saved to: {self.dfs_folder}")

    def _load_feature_mapping(self):
        """Load feature ID to name mapping from JSON file"""
        try:
            mapping_file = os.path.join(os.path.dirname(__file__), '..', 'lsst_features_name_mapping.json')
            with open(mapping_file, 'r') as f:
                # The JSON maps name -> id, we need id -> name
                name_to_id = json.load(f)
                # Invert the mapping to get id -> name
                id_to_name = {v: k for k, v in name_to_id.items()}
                self.logger.info(f"Loaded {len(id_to_name)} feature mappings")
                return id_to_name
        except Exception as e:
            self.logger.error(f"Error loading feature mapping: {e}")
            return {}

    def save_dataframe(self, df, oid=None):
        """Save DataFrame to dfs_staging folder"""
        try:
            if df.empty:
                self.logger.warning("DataFrame is empty, not saving")
                return
            
            # Generate timestamp and random hash
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_hash = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
            
            if oid:
                filename = f"features_{oid}_{timestamp}_{random_hash}.csv"
            else:
                filename = f"features_{timestamp}_{random_hash}.csv"
            
            filepath = os.path.join(self.dfs_folder, filename)
            
            # Save as CSV
            df.to_csv(filepath, index=False)
            self.logger.info(f"DataFrame saved to: {filepath}")
            print(f"DataFrame saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving DataFrame: {e}")

    def save_unique_bands(self, unique_bands):
        """Save unique band values from the batch to a file"""
        try:
            if not unique_bands:
                self.logger.warning("No unique bands to save")
                return
            
            # Generate timestamp and random hash
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_hash = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
            filename = f"unique_bands_batch_{timestamp}_{random_hash}.json"
            filepath = os.path.join(self.dfs_folder, filename)
            
            # Convert set to sorted list for JSON serialization
            bands_data = {
                "timestamp": timestamp,
                "hash": random_hash,
                "unique_bands": sorted(list(unique_bands)),
                "total_unique_bands": len(unique_bands)
            }
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(bands_data, f, indent=2)
            
            self.logger.info(f"Unique bands saved to: {filepath}")
            self.logger.info(f"Found {len(unique_bands)} unique bands: {sorted(list(unique_bands))}")
            print(f"Unique bands saved to: {filepath}")
            print(f"Unique bands found: {sorted(list(unique_bands))}")
            
        except Exception as e:
            self.logger.error(f"Error saving unique bands: {e}")

    def extract_features_to_dataframe(self, message):
        """Extract features from message payload and create DataFrame with feature names"""
        try:
            # Check if message has payload
            if 'payload' in message:
                payload_str = message['payload']
                
                # Parse the JSON string in payload
                if isinstance(payload_str, str):
                    payload = json.loads(payload_str)
                else:
                    payload = payload_str
                
                # Extract oid and features from payload
                oid = None
                if 'payload' in payload:
                    oid = payload['payload'].get('oid')
                    
                if 'payload' in payload and 'features' in payload['payload']:
                    features = payload['payload']['features']
                    
                    # Create DataFrame with feature_id, feature_name, band, value
                    df_data = []
                    for feature in features:
                        feature_id = feature.get('feature_id')
                        feature_name = self.feature_mapping.get(feature_id, f"unknown_feature_{feature_id}")
                        
                        df_data.append({
                            'feature_id': feature_id,
                            'feature_name': feature_name,
                            'band': feature.get('band'),
                            'value': feature.get('value')
                        })
                    
                    df = pd.DataFrame(df_data)
                    self.logger.info(f"Created DataFrame with {len(df)} features for oid: {oid}")
                    #print(f"Features DataFrame:")
                    #print(df.to_string(index=False))
                    
                    # Save DataFrame
                    self.save_dataframe(df, oid)
                    
                    return df
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON payload: {e}")
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
        
        return pd.DataFrame()

    def pre_execute(self, messages: List[dict]):
        """Pre-execute method - no longer uses secondary consumer"""
        self.logger.info("Pre-execute - messages will be processed in execute method")
        
        # Return the original messages unchanged
        return messages

    def execute(self, messages):
        """Execute method - now processes all incoming messages"""
        self.logger.info(f"Executing with {len(messages)} messages")
        
        all_features_dfs = []
        all_unique_bands = set()  # To collect all unique band values
        output = []
        
        for i, message in enumerate(messages):
            self.logger.info(f"Processing message {i+1}/{len(messages)}")
            
            try:
                # Extract features from each message and create DataFrame
                features_df = self.extract_features_to_dataframe(message)
                
                if not features_df.empty:
                    self.logger.info(f"Successfully extracted {len(features_df)} features from message {i+1}")
                    all_features_dfs.append(features_df)
                    
                    # Get unique bands from this DataFrame
                    if 'band' in features_df.columns:
                        df_unique_bands = set(features_df['band'].dropna().unique())
                        all_unique_bands.update(df_unique_bands)
                        self.logger.info(f"Message {i+1} unique bands: {sorted(df_unique_bands)}")
                    
                else:
                    self.logger.warning(f"No features found in message {i+1}")
                
                # Create output for each message
                output.append({
                    "oid": message.get("oid", "unknown"),
                    "processed": True,
                    "survey": self.survey,
                    "message_number": i+1,
                    "features_extracted": len(features_df) if not features_df.empty else 0
                })
                
            except Exception as e:
                self.logger.error(f"Error processing message {i+1}: {e}")
                output.append({
                    "oid": message.get("oid", "unknown"),
                    "processed": False,
                    "survey": self.survey,
                    "message_number": i+1,
                    "error": str(e)
                })
        
        # Save unique bands from all messages in the batch
        self.save_unique_bands(all_unique_bands)
        
        # Log summary of processed messages
        total_features = sum(len(df) for df in all_features_dfs)
        self.logger.info(f"Processed {len(messages)} messages, extracted {total_features} total features")
        self.logger.info(f"Total unique bands found in batch: {sorted(all_unique_bands)}")
        
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