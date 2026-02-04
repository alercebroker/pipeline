from sqlalchemy import select, cast, Float
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from db_plugins.db.sql.models import (
    Detection, ZtfDetection, ZtfForcedPhotometry, 
    ForcedPhotometry, ZtfNonDetection
)
import logging
from ..schemas.schema_applier import apply_schema
from typing import List, Dict, Any
from .database_strategy import DatabaseStrategy
import math

# Constants for ZTF error coordinates calculations
ERRORS = {
    1: 0.065,
    2: 0.085,
    3: 0.01,
}


def _e_ra(dec, fid):
    try:
        return ERRORS[fid] / abs(math.cos(math.radians(dec)))
    except ZeroDivisionError:
        return float("nan")

class ZTFDatabaseStrategy(DatabaseStrategy):
    """
    Database strategy for ZTF survey.
    """
    def __init__(self, db_connection):
        super().__init__(db_connection)
        self._schemas = None

    def _get_schemas(self):
        """Load schemas for ZTF survey"""
        if self._schemas is None:
            from core.schemas.ZTF.ZTF_db_schemas import (
                non_detections_ztf_db, 
                forced_photometry_ztf_db, 
                detections_ztf_db
            )
            self._schemas = {
                'detections': detections_ztf_db,
                'forced_photometry': forced_photometry_ztf_db,
                'non_detections': non_detections_ztf_db
            }
        return self._schemas
    
    def get_detection_schema(self) -> Dict[str, Any]:
        """Return the schema for ZTF detections."""
        return self._get_schemas()['detections']
    
    def get_forced_photometry_schema(self) -> Dict[str, Any]:
        """Return the schema for ZTF forced photometry."""
        return self._get_schemas()['forced_photometry']
    
    def get_non_detection_schema(self) -> Dict[str, Any]:
        """Return the schema for ZTF non-detections."""
        return self._get_schemas()['non_detections']
    
    def get_detections(self, oids: List[str]) -> List[Dict[str, Any]]:
        """
        Get ZTF detections with proper casting for precision fields.
        """
        if not oids:
            return []
            
        oids = [int(oid) for oid in oids]
        
        with self.db_connection.session() as session:
            # Query ZTF-specific detection data with casting
            ztf_stmt = select(
                ZtfDetection.oid,
                ZtfDetection.sid,
                ZtfDetection.measurement_id,
                ZtfDetection.pid,
                cast(ZtfDetection.diffmaglim, DOUBLE_PRECISION).label('diffmaglim'),
                cast(ZtfDetection.magpsf, DOUBLE_PRECISION).label('mag'),
                cast(ZtfDetection.sigmapsf, DOUBLE_PRECISION).label('e_mag'),
                cast(ZtfDetection.magap, DOUBLE_PRECISION).label('magap'),
                cast(ZtfDetection.sigmagap, DOUBLE_PRECISION).label('sigmagap'),
                cast(ZtfDetection.distnr, DOUBLE_PRECISION).label('distnr'),
                cast(ZtfDetection.rb, DOUBLE_PRECISION).label('rb'),
                cast(ZtfDetection.drb, DOUBLE_PRECISION).label('drb'),
                cast(ZtfDetection.magapbig, DOUBLE_PRECISION).label('magapbig'),
                cast(ZtfDetection.sigmagapbig, DOUBLE_PRECISION).label('sigmagapbig'),
                ZtfDetection.isdiffpos,
                ZtfDetection.nid,
                ZtfDetection.rbversion,
                ZtfDetection.drbversion,
                ZtfDetection.rfid,
                ZtfDetection.magpsf_corr,
                ZtfDetection.sigmapsf_corr,
                ZtfDetection.sigmapsf_corr_ext,
                ZtfDetection.corrected,
                ZtfDetection.dubious,
                ZtfDetection.parent_candid,
                ZtfDetection.has_stamp
            ).where(ZtfDetection.oid.in_(oids))

            ztf_detections = session.execute(ztf_stmt).all()
            
            # Query general detection data
            stmt = select(Detection).where(Detection.oid.in_(oids))
            detections = session.execute(stmt).all()
            
            # Parse the results
            return self._parse_ztf_detections(ztf_detections, detections)
    
    def get_forced_photometry(self, oids: List[str]) -> List[Dict[str, Any]]:
        """
        Get ZTF forced photometry with proper casting for precision fields.
        """
        if not oids:
            return []
            
        oids = [int(oid) for oid in oids]
        
        with self.db_connection.session() as session:
            # Query ZTF-specific forced photometry with casting
            ztf_stmt = select(
                ZtfForcedPhotometry.oid,
                ZtfForcedPhotometry.sid,
                ZtfForcedPhotometry.measurement_id,
                ZtfForcedPhotometry.pid,
                ZtfForcedPhotometry.mag,
                ZtfForcedPhotometry.e_mag,
                ZtfForcedPhotometry.mag_corr.label('magpsf_corr'),  # Rename for consistency during correction of magnitudes
                ZtfForcedPhotometry.e_mag_corr.label('sigmapsf_corr'),
                ZtfForcedPhotometry.e_mag_corr_ext.label('sigmapsf_corr_ext'),
                ZtfForcedPhotometry.isdiffpos,
                ZtfForcedPhotometry.corrected,
                ZtfForcedPhotometry.dubious,
                ZtfForcedPhotometry.parent_candid,
                ZtfForcedPhotometry.field,
                ZtfForcedPhotometry.rcid,
                ZtfForcedPhotometry.rfid,
                cast(ZtfForcedPhotometry.sciinpseeing, DOUBLE_PRECISION).label('sciinpseeing'),
                cast(ZtfForcedPhotometry.scibckgnd, DOUBLE_PRECISION).label('scibckgnd'),
                cast(ZtfForcedPhotometry.scisigpix, DOUBLE_PRECISION).label('scisigpix'),
                cast(ZtfForcedPhotometry.magzpsci, DOUBLE_PRECISION).label('magzpsci'),
                cast(ZtfForcedPhotometry.magzpsciunc, DOUBLE_PRECISION).label('magzpsciunc'),
                cast(ZtfForcedPhotometry.magzpscirms, DOUBLE_PRECISION).label('magzpscirms'),
                cast(ZtfForcedPhotometry.clrcoeff, DOUBLE_PRECISION).label('clrcoeff'),
                cast(ZtfForcedPhotometry.clrcounc, DOUBLE_PRECISION).label('clrcounc'),
                cast(ZtfForcedPhotometry.exptime, DOUBLE_PRECISION).label('exptime'),
                cast(ZtfForcedPhotometry.adpctdif1, DOUBLE_PRECISION).label('adpctdif1'),
                cast(ZtfForcedPhotometry.adpctdif2, DOUBLE_PRECISION).label('adpctdif2'),
                cast(ZtfForcedPhotometry.diffmaglim, DOUBLE_PRECISION).label('diffmaglim'),
                ZtfForcedPhotometry.programid,
                ZtfForcedPhotometry.procstatus,
                cast(ZtfForcedPhotometry.distnr, DOUBLE_PRECISION).label('distnr'),
                ZtfForcedPhotometry.ranr,
                ZtfForcedPhotometry.decnr,
                cast(ZtfForcedPhotometry.magnr, DOUBLE_PRECISION).label('magnr'),
                cast(ZtfForcedPhotometry.sigmagnr, DOUBLE_PRECISION).label('sigmagnr'),
                cast(ZtfForcedPhotometry.chinr, DOUBLE_PRECISION).label('chinr'),
                cast(ZtfForcedPhotometry.sharpnr, DOUBLE_PRECISION).label('sharpnr')
                ).where(ZtfForcedPhotometry.oid.in_(oids))
            
            ztf_forced = session.execute(ztf_stmt).all()
            
            # Query general forced photometry
            stmt = select(ForcedPhotometry).where(ForcedPhotometry.oid.in_(oids))
            forced = session.execute(stmt).all()
            
            # Parse the results
            return self._parse_ztf_forced_photometry(ztf_forced, forced)
    
    def get_non_detections(self, oids: List[str]) -> List[Dict[str, Any]]:
        """
        Get ZTF non-detections with proper casting for precision fields.
        """
        if not oids:
            return []
            
        oids = [int(oid) for oid in oids]
        
        with self.db_connection.session() as session:
            # Query with casting for precision fields
            stmt = select(
                ZtfNonDetection.oid,
                ZtfNonDetection.sid,
                ZtfNonDetection.band,
                ZtfNonDetection.mjd,
                cast(ZtfNonDetection.diffmaglim, DOUBLE_PRECISION).label('diffmaglim')
            ).where(ZtfNonDetection.oid.in_(oids))

            non_detections = session.execute(stmt).all()
            
            # Parse the results 
            return self._parse_ztf_non_detections(non_detections)
    
    def _parse_ztf_detections(self, ztf_models: list, models: list) -> List[Dict[str, Any]]:
        """
        Parse ZTF detections with proper handling of cast results.
        """
        parsed_ztf_dets = []

        # Convert Row objects to dictionaries
        for row in ztf_models:
            parsed_det = {}
            for key, value in row._mapping.items():
                parsed_det[key] = value
            parsed_ztf_dets.append(parsed_det)

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
            parsed_d.pop("created_date", None)
            parsed_dets.append(parsed_d)

        det_lookup = {}
        for det_record in parsed_dets:
            key = (det_record['oid'], det_record['measurement_id'])
            det_lookup[key] = det_record

        parsed_ztf_list = []

        # Process ZTF detection records that have a matching detection record
        for ztf_det in parsed_ztf_dets:
            ztf_key = (ztf_det['oid'], ztf_det['measurement_id'])
            
            if ztf_key in det_lookup:
                # Create combined dictionary
                combined_dict = {**ztf_det, **det_lookup[ztf_key]}
                combined_dict["new"] = False
                combined_dict["forced"] = False
                combined_dict["tid"] = 0

                # Calculate e_ra and e_dec. If not possible, set to float('nan') so it keeps the float type
                e_ra = _e_ra(combined_dict["dec"], combined_dict["band"])
                e_dec = ERRORS.get(combined_dict["band"], float("nan"))
                combined_dict["e_ra"] = e_ra
                combined_dict["e_dec"] = e_dec

                # Append to final list
                parsed_ztf_list.append(combined_dict)


        return parsed_ztf_list

    def _parse_ztf_forced_photometry(self, ztf_models: list, models: list) -> List[Dict[str, Any]]:
        """
        Parse ZTF forced photometry with proper handling of cast results.
        """
        parsed_ztf_fphots = []
        
        # Convert Row objects to dictionaries
        for row in ztf_models:
            parsed_fp_d = {}
            for key, value in row._mapping.items():
                parsed_fp_d[key] = value
            parsed_ztf_fphots.append(parsed_fp_d)

        parsed_fphots = []
        for d in models:
            d: dict = d[0].__dict__
            parsed_d = {}
            for field, value in d.items():
                if field.startswith("_"):
                    continue
                else:
                    parsed_d[field] = value
            parsed_d.pop("created_date", None)
            parsed_fphots.append(parsed_d)
            
        fp_lookup = {}
        for fp_record in parsed_fphots:
            key = (fp_record['oid'], fp_record['measurement_id'])
            fp_lookup[key] = fp_record

        parsed_ztf_list = []

        # Process ZTF records that have a matching forced photometry record
        for ztf_record in parsed_ztf_fphots:
            ztf_key = (ztf_record['oid'], ztf_record['measurement_id'])
            
            if ztf_key in fp_lookup:
                # Create combined dictionary
                combined_dict = {**ztf_record, **fp_lookup[ztf_key]}
                combined_dict["new"] = False
                combined_dict["forced"] = True
                combined_dict["tid"] = 0
                
                # Calculate e_ra and e_dec. If not possible, set to float('nan') so it keeps the float type
                e_ra = _e_ra(combined_dict["dec"], combined_dict["band"])
                e_dec = ERRORS.get(combined_dict["band"], float("nan"))
                combined_dict["e_ra"] = e_ra
                combined_dict["e_dec"] = e_dec

                # Append to final list
                parsed_ztf_list.append(combined_dict)

        return parsed_ztf_list
    
    def _parse_ztf_non_detections(self, ztf_models: list) -> List[Dict[str, Any]]:
        """
        Parse ZTF non-detections with proper handling of cast results.
        """
        non_dets = []

        # Convert Row objects to dictionaries
        for row in ztf_models:
            parsed_non_det = {}
            for key, value in row._mapping.items():
                parsed_non_det[key] = value
                parsed_non_det["tid"] = 0
            non_dets.append(parsed_non_det)


        return non_dets
    
    def get_all_historical_data_as_dataframes(self, oids):
        """
        Get all historical data types for given OIDs as DataFrames with schemas applied.
        
        This is the main method that should be called by step.py for processed data for ZTF survey.
        
        Args:
            oids: List of object IDs to query
            
        Returns:
            Dict with DataFrames for each data type, with proper schemas applied
        """
        raw_data = {
            'detections': self.get_detections(oids),
            'forced_photometry': self.get_forced_photometry(oids),
            'non_detections': self.get_non_detections(oids)
        }

        result = {
            'detections': apply_schema(raw_data['detections'], self.get_detection_schema()),
            'forced_photometry': apply_schema(raw_data['forced_photometry'], self.get_forced_photometry_schema()),
            'non_detections': apply_schema(raw_data['non_detections'], self.get_non_detection_schema())
        }

        logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        logger.info(
            f"Retrieved {result['detections'].shape[0]} detections, "
            f"{result['forced_photometry'].shape[0]} forced photometries "
            f"and {result['non_detections'].shape[0]} non_detections from the database"
        )
        return result