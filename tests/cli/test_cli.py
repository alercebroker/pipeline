from click.testing import CliRunner
from cli.alert_archive import cli, download_archive, concat_files, upload_s3
from unittest import mock

from urllib.request import (
    HTTPHandler,
    build_opener,
    install_opener,
)
from urllib.response import addinfourl
from io import BytesIO
import pytest


def mock_response(req):
    data = BytesIO(b"mock data")
    headers = {"Content-Length": data.getbuffer().nbytes}
    resp = addinfourl(data, headers, req.get_full_url())
    resp.code = 200
    resp.msg = "OK"
    return resp


class MockHTTPHandler(HTTPHandler):
    def http_open(self, req):
        return mock_response(req)


mock_opener = build_opener(MockHTTPHandler)
install_opener(mock_opener)


@mock.patch("cli.alert_archive.download_archive")
@mock.patch("cli.alert_archive.concat_files")
@mock.patch("cli.alert_archive.upload_s3")
def test_cli(upload, concat, download):
    runner = CliRunner()
    result = runner.invoke(cli, ["123"])
    download.assert_called()
    upload.assert_called()
    concat.assert_called()
    assert result.exit_code == 0


def test_download_archive():
    runner = CliRunner()
    result = runner.invoke(download_archive, ["http://some-test-data.mock"])
    assert result.exit_code == 0


def test_download_archive_output_dir_error():
    runner = CliRunner()
    result = runner.invoke(
        download_archive,
        [
            "http://some-test-data.mock",
            "--output-dir",
            "something_not_right",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, NotADirectoryError)


def test_download_archive_filename():
    raise NotImplementedError()


def test_download_archive_filename_error():
    raise NotImplementedError()


def test_file_exists_warning():
    raise NotImplementedError()


def test_concat_files():
    raise NotImplementedError()


def test_upload_s3():
    raise NotImplementedError()
