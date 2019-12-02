import jinja2
import argparse
import os
import yaml
import apf

HELPER_PATH = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.abspath(os.path.join(HELPER_PATH,".."))
TEMPLATE_PATH = os.path.abspath(os.path.join(CORE_PATH,"templates"))

def execute_from_command_line(argv):
    parser = argparse.ArgumentParser(description="Alert Processing Framework for astronomy", usage="""apf [-h] command
    newstep         Create package for a new apf step.
    build           Generate Dockerfile and scripts to run a step.
    """)
    parser.add_argument("command", choices=["build", "newstep"])
    parser.add_argument("extras",action='append', nargs="*")
    args,_ = parser.parse_known_args()

    if args.command == "build":
        build_dockerfiles()
    if args.command == "newstep":
        new_step()

def _validate_steps(steps):
    pass

def new_step():
    import re

    parser = argparse.ArgumentParser(description="Create an APF step",
                                    usage="apf newstep [-h] step_name")
    parser.add_argument("newstep")
    parser.add_argument("step_name")
    args,_ = parser.parse_known_args()

    BASE = os.getcwd()

    output_path = os.path.join(BASE,args.step_name)

    if os.path.exists(output_path):
        raise Exception("Output directory already exist.")

    print(f"Creating apf step package in {output_path}")

    #Creating main package directory
    os.makedirs(os.path.join(output_path,args.step_name))
    os.makedirs(os.path.join(output_path,"tests"))
    os.makedirs(os.path.join(output_path,"scripts"))

    loader = jinja2.FileSystemLoader(TEMPLATE_PATH)
    route = jinja2.Environment(loader=loader)


    init_template = route.get_template("step/package/__init__.py")
    with open(os.path.join(output_path,args.step_name,"__init__.py"),"w") as f:
        f.write(init_template.render())

    step_template = route.get_template("step/package/step.py")
    with open(os.path.join(output_path,args.step_name,"step.py"),"w") as f:

        class_name = "".join([word.capitalize() for word in re.split("_|\s|-|\||;|\.|,",args.step_name)])
        f.write(step_template.render(step_name=class_name))

    dockerfile_template = route.get_template("step/Dockerfile.template")
    with open(os.path.join(output_path,"Dockerfile"),"w") as f:
        f.write(dockerfile_template.render())


    run_script_template = route.get_template("step/scripts/run_step.py")
    with open(os.path.join(output_path,"scripts","run_step.py"),"w") as f:
        f.write(run_script_template.render(package_name=args.step_name, class_name=class_name))

    requirements_template = route.get_template("step/requirements.txt")
    with open(os.path.join(output_path,"requirements.txt"),"w") as f:
        dev = os.environ.get("APF_ENVIRONMENT", "production")

        try:
            version = apf.__version__
        except AttributeError:
            version = '0.0.0'
        f.write(requirements_template.render(apf_version=version, develop=(dev=="develop")))

    settings_template = route.get_template("step/settings.py")
    with open(os.path.join(output_path,"settings.py"),"w") as f:
        f.write(settings_template.render(step_name=args.step_name))



def build_dockerfiles():
    parser = argparse.ArgumentParser(description="Create Dockerfiles for steps",
                                    usage="apf build [-h] input output")
    parser.add_argument("build")
    args,_ = parser.parse_known_args()

    pass
