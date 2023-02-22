import jinja2
import os
import apf
import click

HELPER_PATH = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.abspath(os.path.join(HELPER_PATH, ".."))
TEMPLATE_PATH = os.path.abspath(os.path.join(CORE_PATH, "templates"))


@click.group()
def cli():
    pass


def _validate_steps(steps):
    pass


@cli.command()
@click.argument("name")
def new_step(name):
    import re

    BASE = os.getcwd()

    output_path = os.path.join(BASE, name)

    if os.path.exists(output_path):
        raise Exception("Output directory already exist.")

    print(f"Creating apf step package in {output_path}")

    package_name = name.lower()
    cleaned_name = re.split(r"_|\s|-|\||;|\.|,", name)
    # Creating main package directory
    os.makedirs(os.path.join(output_path, package_name))
    os.makedirs(os.path.join(output_path, "tests"))
    os.makedirs(os.path.join(output_path, "scripts"))

    loader = jinja2.FileSystemLoader(TEMPLATE_PATH)
    route = jinja2.Environment(loader=loader)

    init_template = route.get_template("step/package/__init__.py")
    with open(os.path.join(output_path, package_name, "__init__.py"), "w") as f:
        f.write(init_template.render())

    step_template = route.get_template("step/package/step.py")
    with open(os.path.join(output_path, package_name, "step.py"), "w") as f:
        class_name = "".join([word.capitalize() for word in cleaned_name])
        f.write(step_template.render(step_name=class_name))

    dockerfile_template = route.get_template("step/Dockerfile")
    with open(os.path.join(output_path, "Dockerfile"), "w") as f:
        f.write(dockerfile_template.render())

    run_script_template = route.get_template("step/scripts/run_step.py")
    with open(os.path.join(output_path, "scripts", "run_step.py"), "w") as f:
        f.write(
            run_script_template.render(package_name=package_name, class_name=class_name)
        )

    requirements_template = route.get_template("step/requirements.txt")
    with open(os.path.join(output_path, "requirements.txt"), "w") as f:
        try:
            apf_version = apf.__version__
        except AttributeError:
            apf_version = "1.0.0"
        f.write(requirements_template.render(apf_version=apf_version))

    settings_template = route.get_template("step/settings.py")
    with open(os.path.join(output_path, "settings.py"), "w") as f:
        f.write(settings_template.render(step_name=name))


@cli.command()
@click.argument("input")
@click.argument("output")
@click.argument("build")
def build_dockerfiles():
    pass
