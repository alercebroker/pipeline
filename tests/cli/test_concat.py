import tarfile
from cli.core.concat import decompress, concat_avro
from unittest import mock


def test_decompress(tmp_path):
    """Check that a created tar file extracts fine and the content is read"""
    some_file = tmp_path / "some_file.txt"
    some_file.write_text("some text")
    tar_file_path = tmp_path / "tarfile.tar.gz"
    tar = tarfile.open(tar_file_path, "w")
    tar.add(some_file, arcname="some_file.txt")
    tar.close()
    path = decompress(tar_file_path, tmp_path / "extracted")
    path = str(path).split("/")[-1]
    for i in (tmp_path / f"extracted/{path}").iterdir():
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
        output_file = f"partition_{i}.avro"
        name, args, kwargs = kall
        assert str(output_file) in args[0]
