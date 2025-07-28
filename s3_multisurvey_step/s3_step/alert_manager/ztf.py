from .base_manager import BaseAlertManager
import io
from fastavro import writer

class ZTFAlertManager(BaseAlertManager):
    survey_id = "ztf"
    """
    Alert manager for ZTF survey.
    """

    def get_file_name(self, alert: str):
        """
        Returns the file name for the file based on the alerts from the messages.
        """
        reversed_candid = str(alert["candidate"]["candid"])[::-1]
        return f"{reversed_candid}.avro"

    def get_file_data(self, alert: str):
        """
        Returns the file data for the file based on the alerts from the messages.
        """
        avro_io = io.BytesIO()
        writer(avro_io, self.parsed_schema, [alert])
        return avro_io
