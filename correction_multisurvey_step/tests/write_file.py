from sqlalchemy import create_engine, select, text, MetaData
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from contextlib import contextmanager
from typing import ContextManager

from data2 import (
    detections,
    non_detection,
    ztf_forced_photometry,
    forced_photometry,
    objects,
    ztf_detections,
)
from db_plugins.db.sql.models_pipeline import (
    Base,
    Object,
    ZtfDetection,
    ZtfForcedPhotometry,
    ForcedPhotometry,
    NonDetection,
    Detection,
)


class PSQLConnection:
    def __init__(self, config={}):
        self.host = config.get("HOST", "localhost")
        self.user = config.get("USER", "postgres")
        self.password = config.get("PASSWORD", "postgres")
        self.port = config.get("PORT", 5432)
        self.db_name = config.get("DB_NAME", "test_nueva_db")

        self._engine = self._create_engine()
        self._session_factory = scoped_session(
            sessionmaker(autocommit=False, autoflush=True, bind=self._engine)
        )

    def _create_engine(self):
        conn_str = (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
        )
        return create_engine(conn_str)

    @contextmanager
    def session(self):
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self):
        Base.metadata.create_all(self._engine)

    def clear_tables(self):
        """Borra todos los datos de las tablas antes de insertar nuevos datos."""
        with self.session() as session:
            meta = MetaData()
            meta.reflect(bind=self._engine)
            for table in reversed(
                meta.sorted_tables
            ):  # important reverse order to avoid dependency errors.
                session.execute(table.delete())
            session.commit()

    def insert_data(
        self,
        detections_data,
        ztf_detections_data,
        non_detections_data,
        ztf_forced_photometry_data,
        forced_photometry_data,
        objects,
    ):
        with self.session() as session:
            # Insert objects first
            for obj_data in objects:
                obj = Object(**obj_data)
                session.add(obj)
            session.commit()  # commit objects before inserting other data.

            print("Object done!")

            # Insert detections
            for detection_data in detections_data:
                detection = Detection(**detection_data)
                session.add(detection)
            print("Detection listo!")

            # Insert ztf_detections
            for ztf_detection_data in ztf_detections_data:
                ztf_detection = ZtfDetection(**ztf_detection_data)
                session.add(ztf_detection)
            print("ZtfDetection done!")

            # Insert non_detections
            for non_detection_data in non_detections_data:
                non_detection = NonDetection(**non_detection_data)
                session.add(non_detection)
            print("NonDetections done!")

            # Insert ztf_forced_photometry
            if isinstance(ztf_forced_photometry_data, dict):
                forced_phot = ZtfForcedPhotometry(**ztf_forced_photometry_data)
                session.add(forced_phot)
            else:
                for fp_data in ztf_forced_photometry_data:
                    forced_phot = ZtfForcedPhotometry(**fp_data)
                    session.add(forced_phot)
            print("ZTF forced photometry done!")

            # Insert forced_photometry
            if isinstance(forced_photometry_data, dict):
                forced_phot = ForcedPhotometry(**forced_photometry_data)
                session.add(forced_phot)
            else:
                for fp_data in forced_photometry_data:
                    forced_phot = ForcedPhotometry(**fp_data)
                    session.add(forced_phot)
            print("FP done!")


if __name__ == "__main__":
    connection = PSQLConnection()

    try:
        # Corrected connection test
        with connection.session() as session:
            result = session.execute(text("SELECT 1"))
            result.fetchone()
            print("¡Sucesfully connected!")
    except Exception as e:
        print(f"Connection error: {e}")

    connection.create_tables()

    # Delete all data from tables to insert new data
    connection.clear_tables()

    connection.insert_data(
        detections,
        ztf_detections,
        non_detection,
        ztf_forced_photometry,
        forced_photometry,
        objects,
    )

    print("Done! Data inserted.")
