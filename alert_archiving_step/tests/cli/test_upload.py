from cli.core.upload import upload_to_s3
from unittest.mock import patch, call


@patch("cli.core.upload.os.system")
def test_upload_to_s3(system_mock, tmp_path):
    upload_to_s3("test_bucket", "concatenated_files_123456")
    system_mock.assert_called()
    kall = system_mock.mock_calls[0]
    assert kall == call(
        "aws s3 sync concatenated_files_123456 s3://test_bucket/ztf_123456_programid1 --only-show-errors"
    )
