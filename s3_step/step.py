from apf.core.step import GenericStep
import logging
import io
from .s3 import upload_file
import math


class S3Step(GenericStep):
    """S3Step Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)

    def execute(self, message):
        self.logger.info(message["objectId"])
        f = io.BytesIO(self.consumer.message.value())
        year, month, day = self.jd_to_date(message["candidate"]["jd"])
        date = "{}{}{}".format(year, month, int(day))
        upload_file(
            f, date, message["candidate"]["candid"], self.config["STORAGE"]["NAME"])

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
