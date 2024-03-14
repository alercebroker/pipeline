import typer
from typing_extensions import Annotated
from typing import List, Optional

from build import update_packages
from utils import Stage

app = typer.Typer(
    help="""
    A group of commands to update the version of package of the Alerce Pipeline
    """
)


@app.command()
def version(
    packages: List[str],
    version: str = None,
    stage: Stage = Stage.staging,
    dry_run: bool = False,
):
    """
    Update the versions of a list of packages. Can be used with a single package.
    """
    if stage == Stage.staging:
        update_packages(packages, "prerelease", dry_run)
    if stage == Stage.production:
        # si no viene version lanzar error?
        update_packages(packages, version, dry_run)


if __name__ == "__main__":
    app()
