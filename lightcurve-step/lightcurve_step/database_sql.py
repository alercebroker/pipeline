from contextlib import contextmanager
from typing import Callable, ContextManager

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker, Session
from db_plugins.db.sql.models import Detection, NonDetection, ForcedPhotometry


class PSQLConnection:
    def __init__(self, config: dict) -> None:
        url = self.__format_db_url(config)
        self._engine = create_engine(url, echo=True)
        self._session_factory = sessionmaker(
            self._engine,
        )

    def __format_db_url(self, config):
        return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"

    @contextmanager
    def session(self) -> Callable[..., ContextManager[Session]]:
        session: Session = self._session_factory()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()


def default_parser(data, **kwargs):
    return data


def _get_sql_detections(
    oids: dict, db_sql: PSQLConnection, parser: Callable = default_parser
):
    if db_sql is None:
        return []
    with db_sql.session() as session:
        stmt = select(
            Detection.candid,
            Detection.oid,
            Detection.mjd,
            Detection.fid,
            Detection.pid,
            Detection.diffmaglim,
            Detection.isdiffpos,
            Detection.nid,
            Detection.ra,
            Detection.dec,
            Detection.magpsf,
            Detection.sigmapsf,
            Detection.magap,
            Detection.sigmagap,
            Detection.distnr,
            Detection.rb,
            Detection.rbversion,
            Detection.drb,
            Detection.drbversion,
            Detection.magapbig,
            Detection.sigmagapbig,
            Detection.rfid,
            Detection.magpsf_corr,
            Detection.sigmapsf_corr,
            Detection.sigmapsf_corr_ext,
            Detection.parent_candid,
            Detection.has_stamp,
            Detection.step_id_corr
        ).where(Detection.oid.in_(oids))
        detections = session.execute(stmt).all()
        return parser(detections, oids=oids)


def _get_sql_non_detections(oids, db_sql, parser: Callable = default_parser):
    if db_sql is None:
        return []
    with db_sql.session() as session:
        stmt = select(NonDetection).where(NonDetection.oid.in_(oids))
        detections = session.execute(stmt).all()
        return parser(detections, oids=oids)


def _get_sql_forced_photometries(oids, db_sql, parser: Callable = default_parser):
    if db_sql is None:
        return []
    with db_sql.session() as session:
        stmt = select(
            ForcedPhotometry.pid,
            ForcedPhotometry.oid,
            ForcedPhotometry.mjd,
            ForcedPhotometry.fid,
            ForcedPhotometry.ra,
            ForcedPhotometry.dec,
            ForcedPhotometry.e_ra,
            ForcedPhotometry.e_dec,
            ForcedPhotometry.mag,
            ForcedPhotometry.e_mag,
            ForcedPhotometry.isdiffpos,
            ForcedPhotometry.parent_candid,
            ForcedPhotometry.has_stamp,
            # extra fields
            ForcedPhotometry.diffmaglim,
            ForcedPhotometry.pdiffimfilename,
            ForcedPhotometry.programpi,
            ForcedPhotometry.programid,
            ForcedPhotometry.tblid,
            ForcedPhotometry.nid,
            ForcedPhotometry.rcid,
            ForcedPhotometry.field,
            ForcedPhotometry.xpos,
            ForcedPhotometry.ypos,
            ForcedPhotometry.chipsf,
            ForcedPhotometry.magap,
            ForcedPhotometry.sigmagap,
            ForcedPhotometry.distnr,
            ForcedPhotometry.magnr,
            ForcedPhotometry.sigmagnr,
            ForcedPhotometry.chinr,
            ForcedPhotometry.sharpnr,
            ForcedPhotometry.sky,
            ForcedPhotometry.magdiff,
            ForcedPhotometry.fwhm,
            ForcedPhotometry.classtar,
            ForcedPhotometry.mindtoedge,
            ForcedPhotometry.magfromlim,
            ForcedPhotometry.seeratio,
            ForcedPhotometry.aimage,
            ForcedPhotometry.bimage,
            ForcedPhotometry.aimagerat,
            ForcedPhotometry.bimagerat,
            ForcedPhotometry.elong,
            ForcedPhotometry.nneg,
            ForcedPhotometry.nbad,
            ForcedPhotometry.rb,
            ForcedPhotometry.ssdistnr,
            ForcedPhotometry.ssmagnr,
            ForcedPhotometry.ssnamenr,
            ForcedPhotometry.sumrat,
            ForcedPhotometry.magapbig,
            ForcedPhotometry.sigmagapbig,
            ForcedPhotometry.ranr,
            ForcedPhotometry.decnr,
            ForcedPhotometry.sgmag1,
            ForcedPhotometry.srmag1,
            ForcedPhotometry.simag1,
            ForcedPhotometry.szmag1,
            ForcedPhotometry.sgscore1,
            ForcedPhotometry.distpsnr1,
            ForcedPhotometry.ndethist,
            ForcedPhotometry.ncovhist,
            ForcedPhotometry.jdstarthist,
            ForcedPhotometry.jdendhist,
            ForcedPhotometry.scorr,
            ForcedPhotometry.tooflag,
            ForcedPhotometry.objectidps1,
            ForcedPhotometry.objectidps2,
            ForcedPhotometry.sgmag2,
            ForcedPhotometry.srmag2,
            ForcedPhotometry.simag2,
            ForcedPhotometry.szmag2,
            ForcedPhotometry.sgscore2,
            ForcedPhotometry.distpsnr2,
            ForcedPhotometry.objectidps3,
            ForcedPhotometry.sgmag3,
            ForcedPhotometry.srmag3,
            ForcedPhotometry.simag3,
            ForcedPhotometry.szmag3,
            ForcedPhotometry.sgscore3,
            ForcedPhotometry.distpsnr3,
            ForcedPhotometry.nmtchps,
            ForcedPhotometry.rfid,
            ForcedPhotometry.jdstartref,
            ForcedPhotometry.jdendref,
            ForcedPhotometry.nframesref,
            ForcedPhotometry.rbversion,
            ForcedPhotometry.dsnrms,
            ForcedPhotometry.ssnrms,
            ForcedPhotometry.dsdiff,
            ForcedPhotometry.magzpsci,
            ForcedPhotometry.magzpsciunc,
            ForcedPhotometry.magzpscirms,
            ForcedPhotometry.nmatches,
            ForcedPhotometry.clrcoeff,
            ForcedPhotometry.clrcounc,
            ForcedPhotometry.zpclrcov,
            ForcedPhotometry.zpmed,
            ForcedPhotometry.clrmed,
            ForcedPhotometry.clrrms,
            ForcedPhotometry.neargaia,
            ForcedPhotometry.neargaiabright,
            ForcedPhotometry.maggaia,
            ForcedPhotometry.maggaiabright,
            ForcedPhotometry.exptime,
            ForcedPhotometry.drb,
            ForcedPhotometry.drbversion,
            ForcedPhotometry.sciinpseeing,
            ForcedPhotometry.scibckgnd,
            ForcedPhotometry.scisigpix,
            ForcedPhotometry.adpctdif1,
            ForcedPhotometry.adpctdif2,
            ForcedPhotometry.procstatus,
        ).where(ForcedPhotometry.oid.in_(oids))
        forced = session.execute(stmt).all()
        return parser(forced, oids=oids)
