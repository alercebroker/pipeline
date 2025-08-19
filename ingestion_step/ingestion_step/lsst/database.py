import pandas as pd
from db_plugins.db.sql.models import (
    Detection,
    ForcedPhotometry,
    LsstDetection,
    LsstDiaObject,
    LsstForcedPhotometry,
    # LsstNonDetection,
    # LsstSsObject,
    Object,
)
from sqlalchemy.orm import Session

from ingestion_step.core.database import (
    DETECTION_COLUMNS,
    FORCED_DETECTION_COLUMNS,
    OBJECT_COLUMNS,
    db_statement_builder,
)


def insert_dia_objects(session: Session, dia_objects: pd.DataFrame):
    if len(dia_objects) == 0:
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

    objects_sql_stmt = db_statement_builder(Object, objects_dict)
    objects_dia_lsst_sql_stmt = db_statement_builder(
        LsstDiaObject, objects_dia_lsst_dict
    )

    session.execute(objects_sql_stmt)
    session.execute(objects_dia_lsst_sql_stmt)


# def insert_ss_objects(session: Session, ss_objects: pd.DataFrame):
#     if len(ss_objects) == 0:
#         return
#     objects_dict = ss_objects[OBJECT_COLUMNS].to_dict("records")
#     objects_ss_lsst_dict = ss_objects[["oid"]].to_dict("records")
#
#     objects_sql_stmt = db_statement_builder(Object, objects_dict)
#     objects_ss_lsst_sql_stmt = db_statement_builder(LsstSsObject, objects_ss_lsst_dict)
#
#     with driver.session() as session:
#         session.execute(objects_sql_stmt)
#         session.execute(objects_ss_lsst_sql_stmt)
#         session.commit()


def insert_sources(session: Session, sources: pd.DataFrame):
    if len(sources) == 0:
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

    detections_sql_stmt = db_statement_builder(Detection, detections_dict)
    detections_lsst_sql_stmt = db_statement_builder(LsstDetection, detections_lsst_dict)

    session.execute(detections_sql_stmt)
    session.execute(detections_lsst_sql_stmt)


def insert_forced_sources(session: Session, forced_sources: pd.DataFrame):
    if len(forced_sources) == 0:
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

    forced_detections_sql_stmt = db_statement_builder(
        ForcedPhotometry, forced_detections_dict
    )
    forced_detections_lsst_sql_stmt = db_statement_builder(
        LsstForcedPhotometry, forced_detections_lsst_dict
    )

    session.execute(forced_detections_sql_stmt)
    session.execute(forced_detections_lsst_sql_stmt)


# def insert_non_detections(session: Session, non_detections: pd.DataFrame):
#     if len(non_detections) == 0:
#         return
#     non_detections_lsst_dict = non_detections[
#         [
#             "oid",
#             "sid",
#             "ccdVisitId",
#             "band",
#             "mjd",
#             "diaNoise",
#         ]
#     ].to_dict("records")
#
#     non_detections_sql_stmt = db_statement_builder(
#         LsstNonDetection, non_detections_lsst_dict
#     )
#
#     with driver.session() as session:
#         session.execute(non_detections_sql_stmt)
#         session.commit()
