import tarfile
from cli.core.concat import decompress, concat_avro, libpath
from unittest import mock


def test_decompress(tmp_path):
    """Check that a created tar file extracts fine and the content is read"""
    some_file = tmp_path / "some_file.txt"
    some_file.write_text("some text")
    tar_file_path = tmp_path / "tarfile.tar.gz"
    tar = tarfile.open(tar_file_path, "w")
    tar.add(some_file, arcname="some_file.txt")
    tar.close()
    decompress(tar_file_path, tmp_path / "extracted")
    for i in (tmp_path / "extracted").iterdir():
        with open(i, "r") as f:
            assert f.read() == "some text"


@mock.patch("cli.core.concat.os.system")
def test_concat_avro(system_mock, tmp_path):
    for i in range(10):
        some_file = tmp_path / f"file{i}.avro"
        some_file.write_text(f"test{i}")
    concat_avro(tmp_path, tmp_path, partition_size=3)
    system_mock.assert_called()
    kalls = system_mock.mock_calls
    assert len(kalls) == 4
    for i, kall in enumerate(kalls):
        output_file = f"partition_{i}"
        output_file = tmp_path / output_file
        assert (
            mock.call(
                f"java -jar {libpath / 'avro-tools-1.8.2.jar'} concat {tmp_path} {output_file}"
            )
            == kall
        )
