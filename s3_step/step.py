from apf.core.step import GenericStep
import logging
import io
from .s3 import upload_file
import math
from db_plugins.db.sql import SQLConnection
from db_plugins.db.sql.models import Step


class S3Step(GenericStep):
    """S3Step Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(self, consumer=None, config=None, db_connection=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)
        self.db = db_connection or SQLConnection()
        self.db.connect(self.config["DB_CONFIG"]["SQL"])
        self.db.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_VERSION"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["FEATURE_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )

    def execute(self, message):
        self.logger.debug(message["objectId"])
        f = io.BytesIO(self.consumer.messages[0].value())
        year, month, day = self.jd_to_date(message["candidate"]["jd"])
        date = "{}{}{}".format(year, month, int(day))
        upload_file(
            f, date, message["candidate"]["candid"], self.config["STORAGE"]["BUCKET_NAME"])
        
    def jd_to_date(self, jd):
        jd = jd + 0.5
        F, I = math.modf(jd)
        I = int(I)
        A = math.trunc((I - 1867216.25)/36524.25)
        if I > 2299160:
            B = I + 1 + A - math.trunc(A / 4.)
        else:
            B = I
        C = B + 1524
        D = math.trunc((C - 122.1) / 365.25)
        E = math.trunc(365.25 * D)
        G = math.trunc((C - E) / 30.6001)
        day = C - E + F - math.trunc(30.6001 * G)
        if G < 13.5:
            month = G - 1
        else:
            month = G - 13
        if month > 2.5:
            year = D - 4716
        else:
            year = D - 4715
        return year, month, day
