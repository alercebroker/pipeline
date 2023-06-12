import re
import pathlib
import os


def upload_to_s3(bucket, file_dir):
    path = pathlib.Path(file_dir)
    date = re.findall("\d+", str(file_dir))
    if len(date):
        date = date[0]
    else:
        date = ""
    bucket_dir = "s3://{}/ztf_{}_programid1".format(bucket, date)
    command = "aws s3 sync {} {} --only-show-errors".format(path, bucket_dir)
    os.system(command)
