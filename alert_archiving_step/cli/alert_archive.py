import click
from cli.core.download import format_tar_file_url, download
from cli.core.concat import concat_avro, decompress
from cli.core.upload import upload_to_s3


@click.group(invoke_without_command=True)
@click.option("--avro-tools-jar-path", default=None, help="Avro tools utility path")
@click.argument("date")
@click.argument("avro_path")
@click.argument("output_path")
@click.argument("partition_size")
@click.argument("bucket_name")
@click.pass_context
def cli(
    ctx,
    avro_tools_jar_path,
    date,
    avro_path,
    output_path,
    partition_size,
    bucket_name,
):

    if ctx.invoked_subcommand is None:
        tar_file = ctx.invoke(download_archive, date=date, download_dir=avro_path)
        avro_files = ctx.invoke(extract_file, tar_file=tar_file)
        ctx.invoke(
            concat_files,
            avro_tools_jar_path=avro_tools_jar_path,
            avro_path=avro_files,
            output_path=output_path,
            partition_size=partition_size,
        )
        ctx.invoke(upload_s3, bucket_name, output_path)
    else:
        ctx.invoked_subcommand


@cli.command()
@click.option(
    "--ztf-archive-url",
    default="https://ztf.uw.edu/alerts/public/",
    help="ZTF Alert Archive base url",
)
@click.option(
    "--download-dir",
    default=".",
    help="Directory to save data",
)
@click.argument("date")
def download_archive(ztf_archive_url, download_dir, date):
    """Download specific date avro files from ztf archive."""

    url = format_tar_file_url(date, ztf_archive_url)
    return download(url, download_dir)


@cli.command()
@click.argument("tar_file")
@click.option(
    "--output-dir",
    default=None,
    help="Customize extracted files path",
)
def extract_file(tar_file, output_dir):
    return decompress(tar_file, output_dir=output_dir)


@cli.command()
@click.option("--avro-tools-jar-path", default=None, help="Avro tools utility path")
@click.argument("avro_path")
@click.argument("output_path")
@click.argument("partition_size")
def concat_files(avro_tools_jar_path, avro_path, output_path, partition_size):
    """Concat all avro files located on a directory.

    Parameters
    ------------
    tar_file : str
    avro_path : str
        The directory containing all input avro files
    output_path : str
        The directory where avro_files will be concatenated
    partition_size : int
        The size (number of original avro files) of each output concatenated avro file
    """
    concat_avro(avro_path, output_path, int(partition_size), avro_tools_jar_path)


@cli.command()
@click.argument("bucket")
@click.argument("file_dir")
def upload_s3(bucket, file_dir):
    """Upload files from folder to s3."""
    upload_to_s3(bucket, file_dir)


if __name__ == "__main__":
    cli()
