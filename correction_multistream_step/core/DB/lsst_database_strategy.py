from sqlalchemy import select, cast, Float
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from db_plugins.db.sql.models import (
    Detection, LsstDetection, LsstForcedPhotometry, 
    ForcedPhotometry, LsstNonDetection
)
from typing import List, Dict, Any
from .database_strategy import DatabaseStrategy

class LSSTDatabaseStrategy(DatabaseStrategy):
    """
    Database strategy for LSST survey.
    
    Handles all LSST-specific database operations including queries and parsing.
    This consolidates your existing _get_sql_* functions and parse_sql_* functions.
    """
    
    def __init__(self,  db_connection, _schemas=None):
        super().__init__(db_connection)
        self._schemas = None

    def _get_schemas(self):
        """Load schemas for LSST survey"""
        if self._schemas is None:
            from core.schemas.LSST.LSST_db_reduced_schemas import (
                dia_non_detection_lsst_db, 
                dia_forced_sources_lsst_db, 
                dia_source_lsst_db
            )
            self._schemas = {
                'detections': dia_source_lsst_db,
                'forced_photometry': dia_forced_sources_lsst_db,
                'non_detections': dia_non_detection_lsst_db
            }
        return self._schemas
    
    def get_detection_schema(self) -> Dict[str, Any]:
        """Return the schema for LSST detections."""
        return self._get_schemas()['detections']
    
    def get_forced_photometry_schema(self) -> Dict[str, Any]:
        """Return the schema for LSST forced photometry."""
        return self._get_schemas()['forced_photometry']
    
    def get_non_detection_schema(self) -> Dict[str, Any]:
        """Return the schema for LSST non-detections."""
        return self._get_schemas()['non_detections']
    
    def get_detections(self, oids: List[str]) -> List[Dict[str, Any]]:
        """
        Get LSST detections with proper casting for precision fields.
        """
        if not oids:
            return []
            
        oids = [int(oid) for oid in oids]
        
        with self.db_connection.session() as session:
            # Query LSST-specific detection data with casting
            lsst_stmt = select(
                LsstDetection.oid,
                LsstDetection.sid,
                LsstDetection.measurement_id,
                LsstDetection.parentDiaSourceId,
                cast(LsstDetection.psfFlux, DOUBLE_PRECISION).label('psfFlux'),
                cast(LsstDetection.psfFluxErr, DOUBLE_PRECISION).label('psfFluxErr'),
                LsstDetection.psfFlux_flag,
                LsstDetection.psfFlux_flag_edge,
                LsstDetection.psfFlux_flag_noGoodPixels,
                cast(LsstDetection.raErr, DOUBLE_PRECISION).label('raErr'),
                cast(LsstDetection.decErr, DOUBLE_PRECISION).label('decErr')
            ).where(LsstDetection.oid.in_(oids))
            
            lsst_detections = session.execute(lsst_stmt).all()
            
            # Query general detection data
            stmt = select(Detection).where(Detection.oid.in_(oids))
            detections = session.execute(stmt).all()
            
            # Parse the results
            return self._parse_lsst_detections(lsst_detections, detections)
    
    def get_forced_photometry(self, oids: List[str]) -> List[Dict[str, Any]]:
        """
        Get LSST forced photometry with proper casting for precision fields.
        """
        if not oids:
            return []
            
        oids = [int(oid) for oid in oids]
        
        with self.db_connection.session() as session:
            # Query LSST-specific forced photometry with casting
            lsst_stmt = select(
                LsstForcedPhotometry.oid,
                LsstForcedPhotometry.measurement_id,
                LsstForcedPhotometry.sid,
                LsstForcedPhotometry.visit,
                LsstForcedPhotometry.detector,
                cast(LsstForcedPhotometry.psfFlux, DOUBLE_PRECISION).label('psfFlux'),
                cast(LsstForcedPhotometry.psfFluxErr, DOUBLE_PRECISION).label('psfFluxErr')
            ).where(LsstForcedPhotometry.oid.in_(oids))
            
            lsst_forced = session.execute(lsst_stmt).all()
            
            # Query general forced photometry
            stmt = select(ForcedPhotometry).where(ForcedPhotometry.oid.in_(oids))
            forced = session.execute(stmt).all()
            
            # Parse the results
            return self._parse_lsst_forced_photometry(lsst_forced, forced)
    
    def get_non_detections(self, oids: List[str]) -> List[Dict[str, Any]]:
        """
        Get LSST non-detections with proper casting for precision fields.
        """
        if not oids:
            return []
            
        oids = [int(oid) for oid in oids]
        
        with self.db_connection.session() as session:
            # Query with casting for precision fields
            stmt = select(
                LsstNonDetection.oid,
                LsstNonDetection.sid,
                LsstNonDetection.ccdVisitId,
                LsstNonDetection.band,
                LsstNonDetection.mjd,
                cast(LsstNonDetection.diaNoise, DOUBLE_PRECISION).label('diaNoise')
            ).where(LsstNonDetection.oid.in_(oids))
            
            non_detections = session.execute(stmt).all()
            
            # Parse the results 
            return self._parse_lsst_non_detections(non_detections)
    
    def _parse_lsst_detections(self, lsst_models: list, models: list) -> List[Dict[str, Any]]:
        """
        Parse LSST detections with proper handling of cast results.
        """
        parsed_lsst_dets = []

        # Convert Row objects to dictionaries
        for row in lsst_models:
            parsed_det = {}
            for key, value in row._mapping.items():
                parsed_det[key] = value
            parsed_lsst_dets.append(parsed_det)

        parsed_dets = []
        # Handle regular detection models
        for d in models:
            d: dict = d[0].__dict__
            parsed_d = {}
            for field, value in d.items():
                if field.startswith("_"):
                    continue
                else:
                    parsed_d[field] = value
            parsed_dets.append(parsed_d)

        det_lookup = {}
        for det_record in parsed_dets:
            key = (det_record['oid'], det_record['measurement_id'])
            det_lookup[key] = det_record

        parsed_lsst_list = []

        # Process LSST detection records that have a matching detection record
        for lsst_det in parsed_lsst_dets:
            lsst_key = (lsst_det['oid'], lsst_det['measurement_id'])
            
            if lsst_key in det_lookup:
                # Create combined dictionary
                combined_dict = {**lsst_det, **det_lookup[lsst_key]}
                combined_dict["new"] = False
                parsed_lsst_list.append(combined_dict)

        return parsed_lsst_list

    def _parse_lsst_forced_photometry(self, lsst_models: list, models: list) -> List[Dict[str, Any]]:
        """
        Parse LSST forced photometry with proper handling of cast results.
        """
        parsed_lsst_fphots = []
        
        # Convert Row objects to dictionaries
        for row in lsst_models:
            parsed_fp_d = {}
            for key, value in row._mapping.items():
                parsed_fp_d[key] = value
            parsed_lsst_fphots.append(parsed_fp_d)

        parsed_fphots = []
        for d in models:
            d: dict = d[0].__dict__
            parsed_d = {}
            for field, value in d.items():
                if field.startswith("_"):
                    continue
                else:
                    parsed_d[field] = value
            parsed_fphots.append(parsed_d)
            
        fp_lookup = {}
        for fp_record in parsed_fphots:
            key = (fp_record['oid'], fp_record['measurement_id'])
            fp_lookup[key] = fp_record

        parsed_lsst_list = []

        # Process LSST records that have a matching forced photometry record
        for lsst_record in parsed_lsst_fphots:
            lsst_key = (lsst_record['oid'], lsst_record['measurement_id'])
            
            if lsst_key in fp_lookup:
                # Create combined dictionary
                combined_dict = {**lsst_record, **fp_lookup[lsst_key]}
                combined_dict["new"] = False
                parsed_lsst_list.append(combined_dict)

        return parsed_lsst_list

    def _parse_lsst_non_detections(self, lsst_models: list) -> List[Dict[str, Any]]:
        """
        Parse LSST non-detections with proper handling of cast results.
        """
        non_dets = []

        # Convert Row objects to dictionaries
        for row in lsst_models:
            parsed_non_det = {}
            for key, value in row._mapping.items():
                parsed_non_det[key] = value
            non_dets.append(parsed_non_det)


        return non_dets