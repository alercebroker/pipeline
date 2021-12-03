from click.testing import CliRunner
from cli.alert_archive import download_archive
from urllib.request import (
    HTTPSHandler,
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


class MockHTTPHandler(HTTPHandler, HTTPSHandler):
    def https_open(self, req):
        return mock_response(req)

    def http_open(self, req):
        return mock_response(req)


mock_opener = build_opener(MockHTTPHandler)
install_opener(mock_opener)


def test_download_archive(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(download_archive, ["123456"])
        assert result.exit_code == 0
        assert "Downloading: ztf_public_123456.tar.gz" in result.stdout


def test_download_archive_output_dir_error(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            download_archive,
            [
                "123456",
                "--output-dir",
                "something_not_right",
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, NotADirectoryError)


def test_download_archive_filename(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            download_archive,
            [
                "123456",
                "--filename",
                "test_filename",
            ],
        )
        assert result.exit_code == 0


def test_download_archive_filename_error(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            download_archive,
            [
                "123456",
                "--filename",
                "/tmp",
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, IsADirectoryError)


def test_file_exists_warning(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        p = tmp_path / "existing_file"
        p.write_text("this file exists and has content")
        with pytest.warns(
            RuntimeWarning, match="File .* exists, overwritting"
        ):
            result = runner.invoke(
                download_archive,
                [
                    "123456",
                    "--output-dir",
                    tmp_path,
                    "--filename",
                    "existing_file",
                ],
            )
            assert result.exit_code == 0
