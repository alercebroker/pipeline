import os
import logging
import shutil


def upload_to_s3(date, input_dir):
    logger = logging.getLogger("upload_s3")
    # logger.info(f"Root dir:{os.getcwd()}")
    bucket_dir = "s3://ztf-avro/ztf_{}_programid1".format(date)
    logger.info("Uploading avros to s3")
    os.system("aws s3 sync {} {} --only-show-errors".format(input_dir, bucket_dir))
    shutil.rmtree(input_dir)
    logger.info("Avros upload and deleted on disk")
    return
