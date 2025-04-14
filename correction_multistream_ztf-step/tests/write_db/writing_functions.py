from libs.db_plugins_multisurvey.db_plugins.db.sql.models_new import (
    Detection, 
    ZtfDetection, 
    ZtfForcedPhotometry, 
    ForcedPhotometry, 
    NonDetection, 
    Object,
)
from data import (
    detections, 
    ztf_detections, 
    non_detection as non_detections, 
    ztf_forced_photometry, 
    forced_photometry, 
    objects,
)

class InsertData():

    def __init__(self, connection):

        self._engine = connection._engine
        self.session = connection.session

    def insert_data(self):
            with self.session() as session:
                # Insert objects first
                for obj_data in objects:
                    obj = Object(**obj_data)
                    session.add(obj)
                session.commit() # commit objects before inserting other data.
                
                print('Object listo!')

                # Insertar detecciones
                for detection_data in detections:
                    detection = Detection(**detection_data)
                    session.add(detection)
                print('Detection listo!')

                # Insertar ztf_detecciones
                for ztf_detection_data in ztf_detections:
                    ztf_detection = ZtfDetection(**ztf_detection_data)
                    session.add(ztf_detection)
                print('ZtfDetection listo!')
                
                # Insertar no-detecciones
                for non_detection_data in non_detections:
                    non_detection = NonDetection(**non_detection_data)
                    session.add(non_detection)
                print('NonDetection listo!')
                # Insertar fotometría forzada
                if isinstance(ztf_forced_photometry, dict):
                    forced_phot = ZtfForcedPhotometry(**ztf_forced_photometry)
                    session.add(forced_phot)
                else:
                    for fp_data in ztf_forced_photometry:
                        forced_phot = ZtfForcedPhotometry(**fp_data)
                        session.add(forced_phot)
                print('ZTFFP listo!')
                # Insertar fotometría forzada
                if isinstance(forced_photometry, dict):
                    forced_phot = ForcedPhotometry(**forced_photometry)
                    session.add(forced_phot)
                else:
                    for fp_data in forced_photometry:
                        forced_phot = ForcedPhotometry(**fp_data)
                        session.add(forced_phot)
                print('FP listo!')