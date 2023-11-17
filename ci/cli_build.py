import typer
from typing_extensions import Annotated
from typing import List, Optional
import json

from build import build_packages
from utils import Stage

app = typer.Typer(
    help="""
    This help pannel may be
    extensive

    and have multiple parts?

    more?
    """
)

state_file_name = ".build_state.json"

@app.command()
def prepare():
    """
    Prepare the manifesto file.
    """
    empty_dict = {}
    file = open(state_file_name, 'w')
    json.dump(empty_dict, file)
    file.close()

@app.command()
def execute(stage: Stage = Stage.staging, dry_run: bool = False, clear: bool = False):
    """
    Read the manifesto file and execute the deploy with its instruction.
    """
    file = open(state_file_name, 'r+')
    packages_dict = json.load(file)
    #_build(packages_dict, dry_run)
    if clear:
        file.truncate(0)
    file.close()
        
@app.command()
def build_direct(
    package: str,
    package_dir: str = None,
    build_arg: Annotated[Optional[List[str]], typer.Option()] = None,
    stage: Stage = Stage.staging,
    dry_run: bool = False
):
    """
    Build a package.
    """
    build_args_names = []
    build_args_values = []
    for a in build_arg:
        name, value = a.split(':')
        build_args_names.append(name)
        build_args_values.append(value)
        
    package_dict = {}
    package_dict[package] = {
        "build-arg": build_args_names,
        "value": build_args_values,
        "package-dir": package if package_dir is None else package_dir,

    }

    _build(package_dict, dry_run)

@app.command()
def build(
    package: str,
    package_dir: str = None,
    build_arg: Annotated[Optional[List[str]], typer.Option()] = None,
):
    """
    Update the manifesto file to add a package to build
    """
    file = open(state_file_name, '+r')
    packages_dict = json.load(file)
    file.close()

    build_args_names = []
    build_args_values = []
    for a in build_arg:
        name, value = a.split(':')
        build_args_names.append(name)
        build_args_values.append(value)
        
    packages_dict[package] = {
        "build-arg": build_args_names,
        "value": build_args_values,
        "package-dir": package if package_dir is None else package_dir,

    }

    file = open(state_file_name, "w")
    json.dump(packages_dict, file)
    file.close()

def _build(packages: dict, dry_run: bool):
    """
    Call the deploy staging or deploy producction according to stage
    with the packages dict. Essentially a wrapper.
    """
    build_packages(packages, dry_run)

if __name__ == "__main__":
    app()



