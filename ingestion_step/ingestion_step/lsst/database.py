import pandas as pd
from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.sql.models import (
    Detection,
    ForcedPhotometry,
    LsstDetection,
    LsstDiaObject,
    LsstForcedPhotometry,
    LsstNonDetection,
    LsstSsObject,
    Object,
)

from ingestion_step.core.database import (
    DETECTION_COLUMNS,
    FORCED_DETECTION_COLUMNS,
    OBJECT_COLUMNS,
    db_statement_builder,
)

from sqlalchemy.dialects.postgresql import insert as pg_insert


def bulk_insert_on_conflict_do_nothing(session, model, records, conflict_columns=None):
    if not records:
        return
    stmt = pg_insert(model).values(records)
    if conflict_columns:
        stmt = stmt.on_conflict_do_nothing(index_elements=conflict_columns)
    else:
        stmt = stmt.on_conflict_do_nothing()
    session.execute(stmt)


def insert_dia_objects(session, dia_objects: pd.DataFrame):
    if dia_objects.empty:
        return

    objects_dict = dia_objects[OBJECT_COLUMNS].to_dict("records")
    objects_dia_lsst_dict = dia_objects[
        [
            "oid",
            "ra_dec_Cov",
            "pmRa",
            "pmRaErr",
            "pmDec",
            "pmDecErr",
            "parallax",
            "parallaxErr",
            "pmRa_pmDec_Cov",
            "pmRa_parallax_Cov",
            "pmDec_parallax_Cov",
            "pmParallaxLnL",
            "pmParallaxChi2",
            "pmParallaxNdata",
            "u_psfFluxMean",
            "u_psfFluxMeanErr",
            "u_psfFluxSigma",
            "u_psfFluxChi2",
            "u_psfFluxNdata",
            "u_fpFluxMean",
            "u_fpFluxMeanErr",
            "u_fpFluxSigma",
            "g_psfFluxMean",
            "g_psfFluxMeanErr",
            "g_psfFluxSigma",
            "g_psfFluxChi2",
            "g_psfFluxNdata",
            "g_fpFluxMean",
            "g_fpFluxMeanErr",
            "g_fpFluxSigma",
            "r_psfFluxMean",
            "r_psfFluxMeanErr",
            "r_psfFluxSigma",
            "r_psfFluxChi2",
            "r_psfFluxNdata",
            "r_fpFluxMean",
            "r_fpFluxMeanErr",
            "r_fpFluxSigma",
            "i_psfFluxMean",
            "i_psfFluxMeanErr",
            "i_psfFluxSigma",
            "i_psfFluxChi2",
            "i_psfFluxNdata",
            "i_fpFluxMean",
            "i_fpFluxMeanErr",
            "i_fpFluxSigma",
            "z_psfFluxMean",
            "z_psfFluxMeanErr",
            "z_psfFluxSigma",
            "z_psfFluxChi2",
            "z_psfFluxNdata",
            "z_fpFluxMean",
            "z_fpFluxMeanErr",
            "z_fpFluxSigma",
            "y_psfFluxMean",
            "y_psfFluxMeanErr",
            "y_psfFluxSigma",
            "y_psfFluxChi2",
            "y_psfFluxNdata",
            "y_fpFluxMean",
            "y_fpFluxMeanErr",
            "y_fpFluxSigma",
            "nearbyObj1",
            "nearbyObj1Dist",
            "nearbyObj1LnP",
            "nearbyObj2",
            "nearbyObj2Dist",
            "nearbyObj2LnP",
            "nearbyObj3",
            "nearbyObj3Dist",
            "nearbyObj3LnP",
            "u_psfFluxErrMean",
            "g_psfFluxErrMean",
            "r_psfFluxErrMean",
            "i_psfFluxErrMean",
            "z_psfFluxErrMean",
            "y_psfFluxErrMean",
        ]
    ].to_dict("records")

    bulk_insert_on_conflict_do_nothing(session, Object, objects_dict)
    bulk_insert_on_conflict_do_nothing(session, LsstDiaObject, objects_dia_lsst_dict)


def insert_ss_objects(session, ss_objects: pd.DataFrame):
    if ss_objects.empty:
        return

    objects_dict = ss_objects[OBJECT_COLUMNS].to_dict("records")
    objects_ss_lsst_dict = ss_objects[
        [
            "oid",
            "discoverySubmissionDate",
            "firstObservationDate",
            "arc",
            "numObs",
            "MOID",
            "MOIDTrueAnomaly",
            "MOIDEclipticLongitude",
            "MOIDDeltaV",
            "u_H",
            "u_G12",
            "u_HErr",
            "u_G12Err",
            "u_H_u_G12_Cov",
            "u_Chi2",
            "u_Ndata",
            "g_H",
            "g_G12",
            "g_HErr",
            "g_G12Err",
            "g_H_g_G12_Cov",
            "g_Chi2",
            "g_Ndata",
            "r_H",
            "r_G12",
            "r_HErr",
            "r_G12Err",
            "r_H_r_G12_Cov",
            "r_Chi2",
            "r_Ndata",
            "i_H",
            "i_G12",
            "i_HErr",
            "i_G12Err",
            "i_H_i_G12_Cov",
            "i_Chi2",
            "i_Ndata",
            "z_H",
            "z_G12",
            "z_HErr",
            "z_G12Err",
            "z_H_z_G12_Cov",
            "z_Chi2",
            "z_Ndata",
            "y_H",
            "y_G12",
            "y_HErr",
            "y_G12Err",
            "y_H_y_G12_Cov",
            "y_Chi2",
            "y_Ndata",
            "medianExtendedness",
        ]
    ].to_dict("records")

    bulk_insert_on_conflict_do_nothing(session, Object, objects_dict)
    bulk_insert_on_conflict_do_nothing(session, LsstSsObject, objects_ss_lsst_dict)


def insert_sources(session, sources: pd.DataFrame):
    if sources.empty:
        return

    detections_dict = sources[DETECTION_COLUMNS].to_dict("records")
    detections_lsst_dict = sources[
        [
            "oid",
            "sid",
            "measurement_id",
            "parentDiaSourceId",
            "visit",
            "detector",
            "raErr",
            "decErr",
            "ra_dec_Cov",
            "x",
            "xErr",
            "y",
            "yErr",
            "x_y_Cov",
            "centroid_flag",
            "apFlux",
            "apFluxErr",
            "apFlux_flag",
            "apFlux_flag_apertureTruncated",
            "is_negative",
            "snr",
            "psfFlux",
            "psfFluxErr",
            "psfRa",
            "psfRaErr",
            "psfDec",
            "psfDecErr",
            "psfFlux_psfRa_Cov",
            "psfFlux_psfDec_Cov",
            "psfRa_psfDec_Cov",
            "psfLnL",
            "psfChi2",
            "psfNdata",
            "psfFlux_flag",
            "psfFlux_flag_edge",
            "psfFlux_flag_noGoodPixels",
            "trailFlux",
            "trailFluxErr",
            "trailRa",
            "trailRaErr",
            "trailDec",
            "trailDecErr",
            "trailLength",
            "trailLengthErr",
            "trailAngle",
            "trailAngleErr",
            "trailFlux_trailRa_Cov",
            "trailFlux_trailDec_Cov",
            "trailFlux_trailLength_Cov",
            "trailFlux_trailAngle_Cov",
            "trailRa_trailDec_Cov",
            "trailRa_trailLength_Cov",
            "trailRa_trailAngle_Cov",
            "trailDec_trailLength_Cov",
            "trailDec_trailAngle_Cov",
            "trailLength_trailAngle_Cov",
            "trailLnL",
            "trailChi2",
            "trailNdata",
            "trail_flag_edge",
            "dipoleMeanFlux",
            "dipoleMeanFluxErr",
            "dipoleFluxDiff",
            "dipoleFluxDiffErr",
            "dipoleRa",
            "dipoleRaErr",
            "dipoleDec",
            "dipoleDecErr",
            "dipoleLength",
            "dipoleLengthErr",
            "dipoleAngle",
            "dipoleAngleErr",
            "dipoleMeanFlux_dipoleFluxDiff_Cov",
            "dipoleMeanFlux_dipoleRa_Cov",
            "dipoleMeanFlux_dipoleDec_Cov",
            "dipoleMeanFlux_dipoleLength_Cov",
            "dipoleMeanFlux_dipoleAngle_Cov",
            "dipoleFluxDiff_dipoleRa_Cov",
            "dipoleFluxDiff_dipoleDec_Cov",
            "dipoleFluxDiff_dipoleLength_Cov",
            "dipoleFluxDiff_dipoleAngle_Cov",
            "dipoleRa_dipoleDec_Cov",
            "dipoleRa_dipoleLength_Cov",
            "dipoleRa_dipoleAngle_Cov",
            "dipoleDec_dipoleLength_Cov",
            "dipoleDec_dipoleAngle_Cov",
            "dipoleLength_dipoleAngle_Cov",
            "dipoleLnL",
            "dipoleChi2",
            "dipoleNdata",
            "forced_PsfFlux_flag",
            "forced_PsfFlux_flag_edge",
            "forced_PsfFlux_flag_noGoodPixels",
            "snapDiffFlux",
            "snapDiffFluxErr",
            "fpBkgd",
            "fpBkgdErr",
            "ixx",
            "ixxErr",
            "iyy",
            "iyyErr",
            "ixy",
            "ixyErr",
            "ixx_iyy_Cov",
            "ixx_ixy_Cov",
            "iyy_ixy_Cov",
            "ixxPSF",
            "iyyPSF",
            "ixyPSF",
            "shape_flag",
            "shape_flag_no_pixels",
            "shape_flag_not_contained",
            "shape_flag_parent_source",
            "extendedness",
            "reliability",
            "dipoleFitAttempted",
            "pixelFlags",
            "pixelFlags_bad",
            "pixelFlags_cr",
            "pixelFlags_crCenter",
            "pixelFlags_edge",
            "pixelFlags_nodata",
            "pixelFlags_nodataCenter",
            "pixelFlags_interpolated",
            "pixelFlags_interpolatedCenter",
            "pixelFlags_offimage",
            "pixelFlags_saturated",
            "pixelFlags_saturatedCenter",
            "pixelFlags_suspect",
            "pixelFlags_suspectCenter",
            "pixelFlags_streak",
            "pixelFlags_streakCenter",
            "pixelFlags_injected",
            "pixelFlags_injectedCenter",
            "pixelFlags_injected_template",
            "pixelFlags_injected_templateCenter",
        ]
    ].to_dict("records")

    bulk_insert_on_conflict_do_nothing(session, Detection, detections_dict)
    bulk_insert_on_conflict_do_nothing(session, LsstDetection, detections_lsst_dict)


def insert_forced_sources(session, forced_sources: pd.DataFrame):
    if forced_sources.empty:
        return

    forced_detections_dict = forced_sources[FORCED_DETECTION_COLUMNS].to_dict("records")
    forced_detections_lsst_dict = forced_sources[
        [
            "oid",
            "sid",
            "measurement_id",
            "visit",
            "detector",
            "psfFlux",
            "psfFluxErr",
        ]
    ].to_dict("records")

    bulk_insert_on_conflict_do_nothing(session, ForcedPhotometry, forced_detections_dict)
    bulk_insert_on_conflict_do_nothing(session, LsstForcedPhotometry, forced_detections_lsst_dict)


def insert_non_detections(session, non_detections: pd.DataFrame):
    if non_detections.empty:
        return

    non_detections_lsst_dict = non_detections[
        [
            "oid",
            "sid",
            "ccdVisitId",
            "band",
            "mjd",
            "diaNoise",
        ]
    ].to_dict("records")

    bulk_insert_on_conflict_do_nothing(session, LsstNonDetection, non_detections_lsst_dict)