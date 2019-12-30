import jinja2
import argparse
import os
import apf
import sys
from apf.db.sql import Base, models, Session
from apf.db.sql.models import Class
from sqlalchemy import create_engine
import alembic.config
HELPER_PATH = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.abspath(os.path.join(HELPER_PATH, ".."))
TEMPLATE_PATH = os.path.abspath(os.path.join(CORE_PATH, "templates"))
MIGRATIONS_PATH = os.path.abspath(
    os.path.join(CORE_PATH, "../db/sql/"))


def execute_from_command_line(argv):
    parser = argparse.ArgumentParser(description="Alert Processing Framework for astronomy", usage="""apf [-h] command
    newstep         Create package for a new apf step.
    build           Generate Dockerfile and scripts to run a step.
    initdb          Create schema on empty database.
    make_migrations Get differences with database and create migrations
    migrate         Update database from migrations
    """)
    parser.add_argument("command", choices=[
                        "build", "newstep", "initdb", "make_migrations", "migrate"])
    parser.add_argument("extras", action='append', nargs="*")
    args, _ = parser.parse_known_args()

    if args.command == "build":
        build_dockerfiles()
    if args.command == "newstep":
        new_step()
    if args.command == "initdb":
        initdb()
    if args.command == "make_migrations":
        make_migrations()
    if args.command == "migrate":
        migrate()


def _validate_steps(steps):
    pass


def initdb():
    parser = argparse.ArgumentParser(description="Create or connect models to database",
                                     usage="apf initdb [-h] [--settings settings_path]")
    parser.add_argument("--settings", default=os.path.abspath("./settings.py"))
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.settings):
        raise Exception("Settings file not found")
    sys.path.append(os.path.dirname(os.path.expanduser(args.settings)))
    from settings import DB_CONFIG
    if "PSQL" in DB_CONFIG:
        db_config = DB_CONFIG["PSQL"]
        db_credentials = 'postgresql://{}:{}@{}:{}/{}'.format(
            db_config["USER"], db_config["PASSWORD"], db_config["HOST"], db_config["PORT"], db_config["DB_NAME"])
    else:
        db_credentials = "sqlite:///:memory:"
    engine = create_engine(db_credentials)
    Base.metadata.create_all(engine)
    os.chdir(MIGRATIONS_PATH)
    alembic.config.main([
        'stamp', 'head'
    ])
    print("Database created with credentials from {}".format(args.settings))


def make_migrations():
    parser = argparse.ArgumentParser(description="Get differences from database and create migration files",
                                     usage="apf make_migrations [-h] [--settings settings_path]")
    parser.add_argument("--settings", default=os.path.abspath("./settings.py"))
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.settings):
        raise Exception("Settings file not found")
    sys.path.append(os.path.dirname(os.path.expanduser(args.settings)))

    os.chdir(MIGRATIONS_PATH)
    alembicArgs = [
        '--raiseerr',
        'revision', '--autogenerate', '-m', 'tables'
    ]
    alembic.config.main(alembicArgs)


def migrate():
    parser = argparse.ArgumentParser(description="Updates database from migrations",
                                     usage="apf migrate [-h] [--settings settings_path]")
    parser.add_argument("--settings", default=os.path.abspath("./settings.py"))
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.settings):
        raise Exception("Settings file not found")
    sys.path.append(os.path.dirname(os.path.expanduser(args.settings)))

    os.chdir(MIGRATIONS_PATH)
    alembicArgs = [
        '--raiseerr',
        'upgrade', 'head'
    ]
    alembic.config.main(alembicArgs)


def new_step():
    import re

    parser = argparse.ArgumentParser(description="Create an APF step",
                                     usage="apf newstep [-h] step_name")
    parser.add_argument("newstep")
    parser.add_argument("step_name")
    args, _ = parser.parse_known_args()

    BASE = os.getcwd()

    output_path = os.path.join(BASE, args.step_name)

    if os.path.exists(output_path):
        raise Exception("Output directory already exist.")

    print(f"Creating apf step package in {output_path}")

    package_name = args.step_name.lower()
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

        class_name = "".join(
            [word.capitalize() for word in re.split("_|\s|-|\||;|\.|,", args.step_name)])
        f.write(step_template.render(step_name=class_name))

    dockerfile_template = route.get_template("step/Dockerfile.template")
    with open(os.path.join(output_path, "Dockerfile"), "w") as f:
        f.write(dockerfile_template.render())

    run_script_template = route.get_template("step/scripts/run_step.py")
    with open(os.path.join(output_path, "scripts", "run_step.py"), "w") as f:
        f.write(run_script_template.render(
            package_name=package_name, class_name=class_name))

    run_multiprocess_template = route.get_template(
        "step/scripts/run_multiprocess.py")
    with open(os.path.join(output_path, "scripts", "run_multiprocess.py"), "w") as f:
        f.write(run_multiprocess_template.render(
            package_name=package_name, class_name=class_name))

    requirements_template = route.get_template("step/requirements.txt")
    with open(os.path.join(output_path, "requirements.txt"), "w") as f:
        dev = os.environ.get("APF_ENVIRONMENT", "production")

        try:
            version = apf.__version__
        except AttributeError:
            version = '0.0.0'
        f.write(requirements_template.render(
            apf_version=version, develop=(dev == "develop")))

    settings_template = route.get_template("step/settings.py")
    with open(os.path.join(output_path, "settings.py"), "w") as f:
        f.write(settings_template.render(step_name=args.step_name))


def build_dockerfiles():
    parser = argparse.ArgumentParser(description="Create Dockerfiles for steps",
                                     usage="apf build [-h] input output")
    parser.add_argument("build")
    args, _ = parser.parse_known_args()

    pass
