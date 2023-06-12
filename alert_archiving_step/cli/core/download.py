from urllib.request import urlopen
from urllib.parse import urljoin
import pathlib
import warnings


def bar(completed: int):
    bar = "|" + chr(45) * int(completed / 10)
    bar = bar + chr(32) * (10 - int(completed / 10))
    return bar + "|"


def print_status(file_size, file_size_dl):
    status = "%10d  [%3.2f%%] %s" % (
        file_size_dl,
        file_size_dl * 100.0 / file_size,
        "\t",
    )
    status = status + bar(file_size_dl * 100.0 / file_size)
    print(status, end="\r")


def format_total_size(file_size):
    if file_size > 1024 and file_size < 1024 * 1024:
        size = file_size / 1024
        return f"Kb: {size}"

    elif file_size >= 1024 * 1024 and file_size < 1024 * 1024 * 1024:
        size = file_size / (1024 * 1024)
        return f"Mb: {size}"

    elif file_size >= 1024 * 1024 * 1024:
        size = file_size / (1024 * 1024 * 1024)
        return f"Gb: {size}"

    else:
        return f"B: {file_size}"


def format_tar_file_url(date, base_url):
    ztf_file = f"ztf_public_{date}.tar.gz"
    return urljoin(base_url, ztf_file)


def download(url, output_dir):
    file_name = url.split("/")[-1]
    u = urlopen(url)
    path = pathlib.Path(output_dir)
    if not path.is_dir() or not path.exists():
        raise NotADirectoryError(
            "Selected output dir is not a valid directory or doesn't exist"
        )
    path = pathlib.Path(output_dir) / file_name
    if path.is_dir():
        raise IsADirectoryError("The filename is a directory")
    if path.exists():
        warnings.warn(RuntimeWarning(f"File {path} exists, overwritting"))
    f = open(path, "wb")
    meta = u.info()
    file_size = int(meta.get("Content-Length"))
    print("Downloading: {} {}".format(file_name, format_total_size(file_size)))
    file_size_dl = 0
    block_sz = 8192
    print_status(file_size, file_size_dl)
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            print("Finish")
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        print_status(file_size, file_size_dl)

    f.close()
    return path.absolute()
