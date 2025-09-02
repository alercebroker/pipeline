from .base_manager import BaseAlertManager
import io
from fastavro import writer

class LSSTAlertManager(BaseAlertManager):
    survey_id = "lsst"
    """
    Alert manager for LSST survey.
    """
    def get_file_name(self, alert: str):
        """
        Returns the file name for the file based on the alerts from the messages.
        """
        object_id = alert["diaSource"].get("diaObjectId", None)
        measurement_id = alert["diaSource"]["diaSourceId"]

        if not object_id:
            object_id = alert["diaSource"].get("ssObjectId", None)
        
        if not object_id:
            raise ValueError("Object ID is missing in the alert data. Not in dia nor ss")

        return f"{object_id}_{measurement_id}.avro"

    def get_file_data(self, alert: str):
        """
        Returns the file data for the file based on the alerts from the messages.
        """
        avro_io = io.BytesIO()
        avro_data = {
            "cutoutDifference": alert.get("cutoutDifference", None),
            "cutoutScience": alert.get("cutoutScience", None),
            "cutoutTemplate": alert.get("cutoutTemplate", None),
        }
        writer(avro_io, self.parsed_schema, [avro_data])
        avro_io.seek(0)
        return avro_io
