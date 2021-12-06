from unittest import mock
from click.testing import CliRunner
from cli.alert_archive import cli, concat_files, download_archive, upload_s3


@mock.patch("cli.alert_archive.download_archive")
@mock.patch("cli.alert_archive.extract_file")
@mock.patch("cli.alert_archive.concat_files")
@mock.patch("cli.alert_archive.upload_s3")
def test_cli(upload, concat, extract, download):
    runner = CliRunner()
    result = runner.invoke(cli, ["123", ".", ".", "1", "test"])
    download.assert_called()
    extract.assert_called()
    concat.assert_called()
    upload.assert_called()
    assert result.exit_code == 0


@mock.patch("cli.alert_archive.download")
def test_download_archive(download):
    runner = CliRunner()
    result = runner.invoke(
        download_archive,
        [
            "date",
            "--download-dir",
            "output",
        ],
    )
    download.assert_called()
    kall = download.mock_calls[0]
    assert kall == mock.call(
        "https://ztf.uw.edu/alerts/public/ztf_public_date.tar.gz",
        "output",
    )
    assert result.exit_code == 0


@mock.patch("cli.alert_archive.concat_avro")
def test_concat(concat_avro):
    runner = CliRunner()
    result = runner.invoke(
        concat_files,
        ["test", "test", "100", "--avro-tools-jar-path", "test"],
    )
    concat_avro.assert_called()
    kall = concat_avro.mock_calls[0]
    assert kall == mock.call("test", "test", 100, "test")
    assert result.exit_code == 0


@mock.patch("cli.alert_archive.upload_to_s3")
def test_upload(upload_to_s3):
    runner = CliRunner()
    result = runner.invoke(
        upload_s3,
        ["test_bucket", "dir_123456"],
    )
    upload_to_s3.assert_called()
    kall = upload_to_s3.mock_calls[0]
    assert kall == mock.call("test_bucket", "dir_123456")
    assert result.exit_code == 0
