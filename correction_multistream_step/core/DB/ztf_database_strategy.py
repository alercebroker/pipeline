from typing import List, Dict, Any
from .database_strategy import DatabaseStrategy

class ZTFDatabaseStrategy(DatabaseStrategy):
    """
    Database strategy for ZTF survey.
    
    """
    
    def get_detections(self, oids: List[str]) -> List[Dict[str, Any]]:
        """ZTF-specific detection query and parsing."""
        # ZTF + Detection
        pass
    
    def get_forced_photometry(self, oids: List[str]) -> List[Dict[str, Any]]:
        """ZTF-specific forced photometry query and parsing."""
        pass
    
    def get_non_detections(self, oids: List[str]) -> List[Dict[str, Any]]:
        """ZTF-specific non-detection query and parsing."""
        pass