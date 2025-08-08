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
            from core.schemas.LSST.LSST_schemas import (
                dia_forced_sources_lsst_db,
                dia_source_lsst_db,
                dia_non_detection_lsst_db
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
                LsstDetection.visit,
                LsstDetection.detector,
                LsstDetection.parentDiaSourceId,
                cast(LsstDetection.raErr, DOUBLE_PRECISION).label('raErr'),
                cast(LsstDetection.decErr, DOUBLE_PRECISION).label('decErr'),
                cast(LsstDetection.ra_dec_Cov, DOUBLE_PRECISION).label('ra_dec_Cov'),
                cast(LsstDetection.x, DOUBLE_PRECISION).label('x'),
                cast(LsstDetection.xErr, DOUBLE_PRECISION).label('xErr'),
                cast(LsstDetection.y, DOUBLE_PRECISION).label('y'),
                cast(LsstDetection.yErr, DOUBLE_PRECISION).label('yErr'),
                cast(LsstDetection.x_y_Cov, DOUBLE_PRECISION).label('x_y_Cov'),
                LsstDetection.centroid_flag,
                cast(LsstDetection.apFlux, DOUBLE_PRECISION).label('apFlux'),
                cast(LsstDetection.apFluxErr, DOUBLE_PRECISION).label('apFluxErr'),
                LsstDetection.apFlux_flag,
                LsstDetection.apFlux_flag_apertureTruncated,
                LsstDetection.is_negative,
                cast(LsstDetection.snr, DOUBLE_PRECISION).label('snr'),
                cast(LsstDetection.psfFlux, DOUBLE_PRECISION).label('psfFlux'),
                cast(LsstDetection.psfFluxErr, DOUBLE_PRECISION).label('psfFluxErr'),
                cast(LsstDetection.psfRa, DOUBLE_PRECISION).label('psfRa'),
                cast(LsstDetection.psfRaErr, DOUBLE_PRECISION).label('psfRaErr'),
                cast(LsstDetection.psfDec, DOUBLE_PRECISION).label('psfDec'),
                cast(LsstDetection.psfDecErr, DOUBLE_PRECISION).label('psfDecErr'),
                cast(LsstDetection.psfFlux_psfRa_Cov, DOUBLE_PRECISION).label('psfFlux_psfRa_Cov'),
                cast(LsstDetection.psfFlux_psfDec_Cov, DOUBLE_PRECISION).label('psfFlux_psfDec_Cov'),
                cast(LsstDetection.psfRa_psfDec_Cov, DOUBLE_PRECISION).label('psfRa_psfDec_Cov'),
                cast(LsstDetection.psfLnL, DOUBLE_PRECISION).label('psfLnL'),
                cast(LsstDetection.psfChi2, DOUBLE_PRECISION).label('psfChi2'),
                LsstDetection.psfNdata,
                LsstDetection.psfFlux_flag,
                LsstDetection.psfFlux_flag_edge,
                LsstDetection.psfFlux_flag_noGoodPixels,
                cast(LsstDetection.trailFlux, DOUBLE_PRECISION).label('trailFlux'),
                cast(LsstDetection.trailFluxErr, DOUBLE_PRECISION).label('trailFluxErr'),
                cast(LsstDetection.trailRa, DOUBLE_PRECISION).label('trailRa'),
                cast(LsstDetection.trailRaErr, DOUBLE_PRECISION).label('trailRaErr'),
                cast(LsstDetection.trailDec, DOUBLE_PRECISION).label('trailDec'),
                cast(LsstDetection.trailDecErr, DOUBLE_PRECISION).label('trailDecErr'),
                cast(LsstDetection.trailLength, DOUBLE_PRECISION).label('trailLength'),
                cast(LsstDetection.trailLengthErr, DOUBLE_PRECISION).label('trailLengthErr'),
                cast(LsstDetection.trailAngle, DOUBLE_PRECISION).label('trailAngle'),
                cast(LsstDetection.trailAngleErr, DOUBLE_PRECISION).label('trailAngleErr'),
                cast(LsstDetection.trailFlux_trailRa_Cov, DOUBLE_PRECISION).label('trailFlux_trailRa_Cov'),
                cast(LsstDetection.trailFlux_trailDec_Cov, DOUBLE_PRECISION).label('trailFlux_trailDec_Cov'),
                cast(LsstDetection.trailFlux_trailLength_Cov, DOUBLE_PRECISION).label('trailFlux_trailLength_Cov'),
                cast(LsstDetection.trailFlux_trailAngle_Cov, DOUBLE_PRECISION).label('trailFlux_trailAngle_Cov'),
                cast(LsstDetection.trailRa_trailDec_Cov, DOUBLE_PRECISION).label('trailRa_trailDec_Cov'),
                cast(LsstDetection.trailRa_trailLength_Cov, DOUBLE_PRECISION).label('trailRa_trailLength_Cov'),
                cast(LsstDetection.trailRa_trailAngle_Cov, DOUBLE_PRECISION).label('trailRa_trailAngle_Cov'),
                cast(LsstDetection.trailDec_trailLength_Cov, DOUBLE_PRECISION).label('trailDec_trailLength_Cov'),
                cast(LsstDetection.trailDec_trailAngle_Cov, DOUBLE_PRECISION).label('trailDec_trailAngle_Cov'),
                cast(LsstDetection.trailLength_trailAngle_Cov, DOUBLE_PRECISION).label('trailLength_trailAngle_Cov'),
                cast(LsstDetection.trailLnL, DOUBLE_PRECISION).label('trailLnL'),
                cast(LsstDetection.trailChi2, DOUBLE_PRECISION).label('trailChi2'),
                LsstDetection.trailNdata,
                LsstDetection.trail_flag_edge,
                cast(LsstDetection.dipoleMeanFlux, DOUBLE_PRECISION).label('dipoleMeanFlux'),
                cast(LsstDetection.dipoleMeanFluxErr, DOUBLE_PRECISION).label('dipoleMeanFluxErr'),
                cast(LsstDetection.dipoleFluxDiff, DOUBLE_PRECISION).label('dipoleFluxDiff'),
                cast(LsstDetection.dipoleFluxDiffErr, DOUBLE_PRECISION).label('dipoleFluxDiffErr'),
                cast(LsstDetection.dipoleRa, DOUBLE_PRECISION).label('dipoleRa'),
                cast(LsstDetection.dipoleRaErr, DOUBLE_PRECISION).label('dipoleRaErr'),
                cast(LsstDetection.dipoleDec, DOUBLE_PRECISION).label('dipoleDec'),
                cast(LsstDetection.dipoleDecErr, DOUBLE_PRECISION).label('dipoleDecErr'),
                cast(LsstDetection.dipoleLength, DOUBLE_PRECISION).label('dipoleLength'),
                cast(LsstDetection.dipoleLengthErr, DOUBLE_PRECISION).label('dipoleLengthErr'),
                cast(LsstDetection.dipoleAngle, DOUBLE_PRECISION).label('dipoleAngle'),
                cast(LsstDetection.dipoleAngleErr, DOUBLE_PRECISION).label('dipoleAngleErr'),
                cast(LsstDetection.dipoleMeanFlux_dipoleFluxDiff_Cov, DOUBLE_PRECISION).label('dipoleMeanFlux_dipoleFluxDiff_Cov'),
                cast(LsstDetection.dipoleMeanFlux_dipoleRa_Cov, DOUBLE_PRECISION).label('dipoleMeanFlux_dipoleRa_Cov'),
                cast(LsstDetection.dipoleMeanFlux_dipoleDec_Cov, DOUBLE_PRECISION).label('dipoleMeanFlux_dipoleDec_Cov'),
                cast(LsstDetection.dipoleMeanFlux_dipoleLength_Cov, DOUBLE_PRECISION).label('dipoleMeanFlux_dipoleLength_Cov'),
                cast(LsstDetection.dipoleMeanFlux_dipoleAngle_Cov, DOUBLE_PRECISION).label('dipoleMeanFlux_dipoleAngle_Cov'),
                cast(LsstDetection.dipoleFluxDiff_dipoleRa_Cov, DOUBLE_PRECISION).label('dipoleFluxDiff_dipoleRa_Cov'),
                cast(LsstDetection.dipoleFluxDiff_dipoleDec_Cov, DOUBLE_PRECISION).label('dipoleFluxDiff_dipoleDec_Cov'),
                cast(LsstDetection.dipoleFluxDiff_dipoleLength_Cov, DOUBLE_PRECISION).label('dipoleFluxDiff_dipoleLength_Cov'),
                cast(LsstDetection.dipoleFluxDiff_dipoleAngle_Cov, DOUBLE_PRECISION).label('dipoleFluxDiff_dipoleAngle_Cov'),
                cast(LsstDetection.dipoleRa_dipoleDec_Cov, DOUBLE_PRECISION).label('dipoleRa_dipoleDec_Cov'),
                cast(LsstDetection.dipoleRa_dipoleLength_Cov, DOUBLE_PRECISION).label('dipoleRa_dipoleLength_Cov'),
                cast(LsstDetection.dipoleRa_dipoleAngle_Cov, DOUBLE_PRECISION).label('dipoleRa_dipoleAngle_Cov'),
                cast(LsstDetection.dipoleDec_dipoleLength_Cov, DOUBLE_PRECISION).label('dipoleDec_dipoleLength_Cov'),
                cast(LsstDetection.dipoleDec_dipoleAngle_Cov, DOUBLE_PRECISION).label('dipoleDec_dipoleAngle_Cov'),
                cast(LsstDetection.dipoleLength_dipoleAngle_Cov, DOUBLE_PRECISION).label('dipoleLength_dipoleAngle_Cov'),
                cast(LsstDetection.dipoleLnL, DOUBLE_PRECISION).label('dipoleLnL'),
                cast(LsstDetection.dipoleChi2, DOUBLE_PRECISION).label('dipoleChi2'),
                LsstDetection.dipoleNdata,
                LsstDetection.forced_PsfFlux_flag,
                LsstDetection.forced_PsfFlux_flag_edge,
                LsstDetection.forced_PsfFlux_flag_noGoodPixels,
                cast(LsstDetection.snapDiffFlux, DOUBLE_PRECISION).label('snapDiffFlux'),
                cast(LsstDetection.snapDiffFluxErr, DOUBLE_PRECISION).label('snapDiffFluxErr'),
                cast(LsstDetection.fpBkgd, DOUBLE_PRECISION).label('fpBkgd'),
                cast(LsstDetection.fpBkgdErr, DOUBLE_PRECISION).label('fpBkgdErr'),
                cast(LsstDetection.ixx, DOUBLE_PRECISION).label('ixx'),
                cast(LsstDetection.ixxErr, DOUBLE_PRECISION).label('ixxErr'),
                cast(LsstDetection.iyy, DOUBLE_PRECISION).label('iyy'),
                cast(LsstDetection.iyyErr, DOUBLE_PRECISION).label('iyyErr'),
                cast(LsstDetection.ixy, DOUBLE_PRECISION).label('ixy'),
                cast(LsstDetection.ixyErr, DOUBLE_PRECISION).label('ixyErr'),
                cast(LsstDetection.ixx_iyy_Cov, DOUBLE_PRECISION).label('ixx_iyy_Cov'),
                cast(LsstDetection.ixx_ixy_Cov, DOUBLE_PRECISION).label('ixx_ixy_Cov'),
                cast(LsstDetection.iyy_ixy_Cov, DOUBLE_PRECISION).label('iyy_ixy_Cov'),
                cast(LsstDetection.ixxPSF, DOUBLE_PRECISION).label('ixxPSF'),
                cast(LsstDetection.iyyPSF, DOUBLE_PRECISION).label('iyyPSF'),
                cast(LsstDetection.ixyPSF, DOUBLE_PRECISION).label('ixyPSF'),
                LsstDetection.shape_flag,
                LsstDetection.shape_flag_no_pixels,
                LsstDetection.shape_flag_not_contained,
                LsstDetection.shape_flag_parent_source,
                cast(LsstDetection.extendedness, DOUBLE_PRECISION).label('extendedness'),
                cast(LsstDetection.reliability, DOUBLE_PRECISION).label('reliability'),
                LsstDetection.dipoleFitAttempted,
                LsstDetection.pixelFlags,
                LsstDetection.pixelFlags_bad,
                LsstDetection.pixelFlags_cr,
                LsstDetection.pixelFlags_crCenter,
                LsstDetection.pixelFlags_edge,
                LsstDetection.pixelFlags_nodata,
                LsstDetection.pixelFlags_nodataCenter,
                LsstDetection.pixelFlags_interpolated,
                LsstDetection.pixelFlags_interpolatedCenter,
                LsstDetection.pixelFlags_offimage,
                LsstDetection.pixelFlags_saturated,
                LsstDetection.pixelFlags_saturatedCenter,
                LsstDetection.pixelFlags_suspect,
                LsstDetection.pixelFlags_suspectCenter,
                LsstDetection.pixelFlags_streak,
                LsstDetection.pixelFlags_streakCenter,
                LsstDetection.pixelFlags_injected,
                LsstDetection.pixelFlags_injectedCenter,
                LsstDetection.pixelFlags_injected_template,
                LsstDetection.pixelFlags_injected_templateCenter
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