from lightcurve_step.database_sql import _get_sql_detections
from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.sql.models import Detection, Object


def test_get_detections(psql_conn: PsqlDatabase):
    obj = Object(
        oid="1",
        ndethist=1,
        ncovhist=1,
        mjdstarthist=1,
        mjdendhist=1,
        corrected=True,
        stellar=True,
        ndet=1,
        g_r_max=1,
        g_r_max_corr=1,
        g_r_mean=1,
        g_r_mean_corr=1,
        meanra=1,
        meandec=1,
        sigmara=1,
        sigmadec=1,
        deltajd=1,
        firstmjd=1,
        lastmjd=1,
        step_id_corr="1",
        diffpos=True,
        reference_change=True,
    )
    det = Detection(
        candid=1,
        oid="1",
        mjd=1,
        fid=1,
        pid=1,
        diffmaglim=1,
        isdiffpos=1,
        nid=1,
        ra=1,
        dec=1,
        magpsf=1,
        sigmapsf=1,
        magap=1,
        sigmagap=1,
        distnr=1,
        rb=1,
        rbversion="1",
        drb=1,
        drbversion="1",
        magapbig=1,
        sigmagapbig=1,
        rfid=1,
        magpsf_corr=1,
        sigmapsf_corr=1,
        sigmapsf_corr_ext=1,
        corrected=True,
        dubious=True,
        parent_candid=1,
        has_stamp=True,
        step_id_corr="1",
    )
    with psql_conn.session() as session:
        session.add(obj)
        session.commit()
        session.add(det)
        session.commit()
    detections = _get_sql_detections(["1"], psql_conn)
    assert len(detections) == 1
