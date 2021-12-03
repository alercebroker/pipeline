import click
from cli.core.download import download, format_tar_file_url
from cli.core.concat import concat_avro


@click.group(chain=True, invoke_without_command=True)
@click.option("--date-format", default="%Y%m%d", help="")
@click.option(
    "--avro-tools-jar-path", default=None, help="Avro tools utility path"
)
@click.argument("date")
@click.argument("avro_path")
@click.argument("output_path")
@click.argument("partition_size")
@click.pass_context
def cli(
    ctx,
    date_format,
    avro_tools_jar_path,
    date,
    avro_path,
    output_path,
    partition_size,
):
    ctx.invoke(download_archive, date_format=date_format, date=date)
    ctx.invoke(
        concat_files,
        avro_tools_jar_path,
        avro_path,
        output_path,
        partition_size,
    )
    ctx.invoke(upload_s3)


@cli.command()
@click.option(
    "--ztf-archive-url",
    default="https://ztf.uw.edu/alerts/public/",
    help="ZTF Alert Archive base url",
)
@click.option(
    "--output-dir",
    default=".",
    help="Directory to save data",
)
@click.option(
    "--filename",
    default=None,
    help="Output filename",
)
@click.argument("date")
def download_archive(ztf_archive_url, output_dir, filename, date):
    """Download specific date avro files from ztf archive."""

    url = format_tar_file_url(date, ztf_archive_url)
    download(url, output_dir, filename=filename)


@cli.command()
@click.option(
    "--avro-tools-jar-path", default=None, help="Avro tools utility path"
)
@click.argument("avro_path")
@click.argument("output_path")
@click.argument("partition_size")
def concat_files(avro_tools_jar_path, avro_path, output_path, partition_size):
    """Concat all avro files located on a directory.

    Parameters
    ------------
    avro_path : str
        The directory containing all input avro files
    output_path : str
        The directory where avro_files will be concatenated
    partition_size : int
        The size (number of original avro files) of each output concatenated avro file
    """
    print(avro_tools_jar_path)
    print(avro_path)
    print(output_path)
    print(partition_size)
    concat_avro(
        avro_path, output_path, int(partition_size), avro_tools_jar_path
    )


@cli.command()
def upload_s3():
    """Upload files from folder to s3."""
    click.echo("upload")


if __name__ == "__main__":
    cli()
