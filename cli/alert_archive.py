import click
import optparse
from cli.core.download import download, format_tar_file_url


@click.group(chain=True, invoke_without_command=True)
@click.option("--date-format", default="%Y%m%d", help="")
@click.argument("date")
@click.pass_context
def cli(ctx, date_format, date):
    ctx.invoke(download_archive, date_format=date_format, date=date)
    ctx.invoke(concat_files)
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
    url = format_tar_file_url(date, ztf_archive_url)
    download(url, output_dir, filename=filename)


@cli.command("concat")
def concat_files():
    click.echo("concat")


@cli.command("upload")
def upload_s3():
    click.echo("upload")


if __name__ == "__main__":
    cli()
