import typer
from typing_extensions import Annotated
from typing import List, Optional

from build import update_packages
from utils import Stage

app = typer.Typer(
    help="""
    This help pannel may be
    extensive

    and have multiple parts?

    more?
    """
)

@app.command()
def update_version(
    packages: List[str],
    version: str = None,
    stage: Stage = Stage.staging,
    dry_run: bool = False
):
    """
    """
    if stage == Stage.staging:
        update_packages(packages, [], "prerelease", dry_run)
    if stage == Stage.production:
        # si no viene version lanzar error?
        update_packages(packages, [], version, dry_run)
