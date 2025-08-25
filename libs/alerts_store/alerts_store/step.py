from apf.core import get_class
from apf.core.step import GenericStep
import logging
import os
import json
import gzip
from datetime import datetime
from pathlib import Path

class AlertStore(GenericStep):
    
    def __init__(self, config=None, **step_args):
        super().__init__(config=config)
        self.base_folder = config.get("BASE_FOLDER")
        self.compress = config.get("COMPRESS_AVRO", True)
        self.logger = logging.getLogger(__name__)

    def execute(self, messages):
        processed_messages = []
        
        for message in messages:
            try:
                timestamp = self._extract_timestamp(message)
                
                folder_path = self._create_folder_structure(timestamp)
                
                file_path = self._save_message(message, folder_path, timestamp)
                
                self.logger.info(f"Saved message on: {file_path}")
                processed_messages.append(message)
                
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                continue
        
        return processed_messages
    
    def _extract_timestamp(self, message):
        return message['timestamp']
    
    def _create_folder_structure(self, timestamp):
        date_str = timestamp.strftime("%Y/%m/%d")
        folder_path = Path(self.base_folder) / date_str
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
    
    def _save_message(self, message, folder_path, timestamp):
        filename = f"alert_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
        file_path = folder_path / filename
        
        message_data = self._serialize_message(message)
        
        if self.compress:
            file_path = file_path.with_suffix('.json.gz')
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                json.dump(message_data, f, ensure_ascii=False, indent=2)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(message_data, f, ensure_ascii=False, indent=2)
        
        return file_path