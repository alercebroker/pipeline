import typer
import json

from deploy import deploy_staging, deploy_production
from utils import Stage

app = typer.Typer(
    help="""
    Group of commands to deploy packages of the Alerce pipeline.

    There are two ways to deploy packages.
    
    1- A single deploy accion calling the direct command.
        eg: python main.py deploy direct package_name

    2- A bulk deploy acction, this involved 3 commands: prepare,
       add_package and execute.
        eg: 
        poetry run python main.py deploy prepare
        poetry run python main.py deploy add-package package_name_1
        poetry run python main.py deploy add-package package_name_2
        poetry run python main.py deploy execute
    """
)

state_file_name = ".deploy_state.json"

@app.command()
def prepare():
    """
    Start the bulk deploy proccess by seting up the environment.
    """
    empty_dict = {}
    file = open(state_file_name, 'w')
    json.dump(empty_dict, file)
    file.close()

@app.command()
def execute(stage: Stage = Stage.staging, dry_run: bool = False, clear: bool = False):
    """
    Executes the bulk deploy process according to the environment previously setted up.
    """
    file = open(state_file_name, 'r+')
    packages_dict = json.load(file)
    _deploy(packages_dict, stage, dry_run)
    if clear:
        file.truncate(0)
    file.close()
        
@app.command()
def direct(
    package: str,
    chart_folder: str = None,
    values: str = None,
    chart: str = None,
    stage: Stage = Stage.staging,
    dry_run: bool = False
):
    """
    Deploy a single package.
    """
    package_dict = {}
    package_dict[package] = {
        "chart": package if chart is None else chart,
        "values": package if values is None else values,
        "chart-folder": package if chart_folder is None else chart_folder,
    }

    _deploy(package_dict, stage, dry_run)

@app.command()
def add_package(
    package: str,
    chart_folder: str = None,
    values: str = None,
    chart: str = None,
):
    """
    Update the environment setup to add a package to be deployed in bulk
    with other packages.
    """
    file = open(state_file_name, '+r')
    packages_dict = json.load(file)
    file.close()

    packages_dict[package] = {
        "chart": package if chart is None else chart,
        "values": package if values is None else values,
        "chart-folder": package if chart_folder is None else chart_folder,
    }

    file = open(state_file_name, "w")
    json.dump(packages_dict, file)
    file.close()

def _deploy(packages: dict, stage: Stage, dry_run: bool):
    """
    Call the deploy staging or deploy producction according to stage
    with the packages dict. Essentially a wrapper.
    """
    if stage == Stage.staging:
        deploy_staging(packages, dry_run)
    elif stage == Stage.production:
        deploy_production(packages, dry_run)

if __name__ == "__main__":
    app()
