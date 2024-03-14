import typer
from typing_extensions import Annotated
from typing import List, Optional
import json

from build import build_packages
from utils import Stage

app = typer.Typer(
    help="""
    Group of commands to build packages of the Alerce pipeline.

    There are two ways to build packages.
    
    1- A single build accion calling the direct command.
        eg: python main.py build direct package_name

    2- A bulk build acction, this involved 3 commands: prepare,
       add_package and execute.
        eg: 
        poetry run python main.py build prepare
        poetry run python main.py build add-package package_name_1
        poetry run python main.py build add-package package_name_2
        poetry run python main.py build execute

    """
)

state_file_name = ".build_state.json"


@app.command()
def prepare():
    """
    Start the bulk build proccess by seting up the environment.
    """
    empty_dict = {}
    file = open(state_file_name, "w")
    json.dump(empty_dict, file)
    file.close()


@app.command()
def execute(
    stage: Stage = Stage.staging, dry_run: bool = False, clear: bool = False
):
    """
    Executes the bulk build process according to the environment previously setted up.
    """
    file = open(state_file_name, "r+")
    packages_dict = json.load(file)
    _build(packages_dict, dry_run)
    if clear:
        file.truncate(0)
    file.close()


@app.command()
def direct(
    package: str,
    package_dir: str = None,
    build_args: Annotated[Optional[List[str]], typer.Option()] = None,
    stage: Stage = Stage.staging,
    dry_run: bool = False,
):
    """
    Builds a single package.
    """
    build_args_names = []
    build_args_values = []
    for a in build_args:
        name, value = a.split(":")
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
def add_package(
    package: str,
    package_dir: str = None,
    build_args: Annotated[Optional[List[str]], typer.Option()] = None,
):
    """
    Update the environment setup to add a package to be deployed in bulk
    with other packages.
    """
    file = open(state_file_name, "+r")
    packages_dict = json.load(file)
    file.close()

    build_args_names = []
    build_args_values = []
    for a in build_args:
        name, value = a.split(":")
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
