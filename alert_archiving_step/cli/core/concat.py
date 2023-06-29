import tarfile
import pathlib
import os
import re

libpath = pathlib.Path(pathlib.Path(__file__).parent.parent.absolute() / "lib")


def decompress(file_path: pathlib.Path, output_dir: str = None):
    tar = tarfile.open(file_path)
    date = re.findall("\d+", str(file_path))
    if len(date):
        date = date[0]
    else:
        date = ""
    path = output_dir or pathlib.Path(file_path).parent.absolute()
    path = pathlib.Path(path) / f"avro_files_{date}/"
    # if not path.exists():
    #     path.mkdir()
    tar.extractall(path)
    return path.absolute()


def ceil(x):
    low = int(x)
    up = int(x) + 1
    if x - low < up - x:
        return low
    return up


def split_files(files: list, n: int):
    i = 0
    splitted = []
    while i <= len(files):
        limit = i + n
        splitted.append(files[i:limit])
        i = limit
    return splitted


def concat_avro(
    avro_path: str,
    output_path: str,
    partition_size: int,
    avro_tools_jar_path: str = None,
):
    """Concatenate AVRO files in chunks

    Parameters
    --------------
    avro_path : str
        Path to directory with all avro files to be concatenated
    output_path : str
        Path to output directory of concatenated avro files
    partition_size : int
        Size (number of original avro files) of each concatenated avro file
    avro_tools_jar_path : str
        Path to jar executable tool to concat data - opional - default None
    """
    avro_path = pathlib.Path(avro_path)
    date = re.findall("\d+", str(avro_path))
    if len(date):
        date = date[0]
    else:
        date = ""
    output_path = pathlib.Path(output_path) / f"concatenated_avro_files_{date}/"
    avro_tools_jar_path = avro_tools_jar_path or libpath / "avro-tools-1.8.2.jar"
    files = list(avro_path.iterdir())
    n_partitions = ceil(len(files) / partition_size)
    partition_files = split_files(files, n_partitions)
    for i, part in enumerate(partition_files):
        part = [str(p) for p in part]
        output_file = output_path / f"partition_{i}.avro"
        command = (
            f"java -jar {avro_tools_jar_path} concat {' '.join(part)} {output_file}"
        )
        os.system(command)
