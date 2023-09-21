from sqlalchemy.orm import Session
from db_plugins.db.sql.models import NonDetection

def upsert_non_detections(session: Session, table = NonDetection, data=[]):
    pass

def insert(session: Session, table, data=[]):
    pass

def update_generic(session: Session, table, data=[]):
    pass

