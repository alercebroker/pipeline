import sys
from build import build_packages, update_packages
from deploy import deploy_staging, deploy_production


def update_versions(stage, packages, libs, version, dry_run=False):
    if stage == "staging":
        version = "prerelease"  # last check for the upgate version to be prerelease in staging
        update_packages(packages, libs, version, dry_run)
    elif stage == "production":
        update_packages(packages, libs, version, dry_run)
    else:
        raise ValueError(
            f'Invalid stage "{stage}". Valid stages are: staging, production'
        )


def build(stage, packages, dry_run=False):
    if stage == "staging":
        build_packages(packages, dry_run)
    elif stage == "production":
        build_packages(packages, dry_run)
    else:
        raise ValueError(
            f'Invalid stage "{stage}". Valid stages are: staging, production'
        )


def deploy(stage, packages, dry_run=False):
    if stage == "staging":
        deploy_staging(packages, dry_run)
    elif stage == "production":
        deploy_production(packages, dry_run)
    else:
        raise ValueError(
            f'Invalid stage "{stage}". Valid stages are: staging, production'
        )


def parse_build_command(packages):
    """Parse packages and build-args from command line arguments

    Args:
        packages (list): List of packages and build-args
            Example: "output_package1 --build-arg arg1:val1 --build-arg arg2:val2 --package-dir package1"
    Returns:
        dict: Dictionary of packages and build-args
            Example: { "output_package1": { "build-arg": ["arg1", "arg2"], "value": ["val1", "val2"], "package-dir": "package1" } }
    """
    parsed_args = {}
    current_arg = None
    for pkg in packages:
        if pkg.startswith("--build-arg"):
            build_arg = pkg.split("=")[1]
            parsed_args[current_arg]["build-arg"].append(
                build_arg.split(":")[0]
            )
            parsed_args[current_arg]["value"].append(build_arg.split(":")[1])
        elif pkg.startswith("--package-dir"):
            package_dir = pkg.split("=")[1]
            parsed_args[current_arg]["package-dir"] = package_dir
        else:
            current_arg = pkg
            parsed_args[pkg] = {
                "build-arg": [],
                "value": [],
                "package-dir": pkg,
            }
    return parsed_args


def parse_deploy_command(releases):
    """Parse packages and build-args from command line arguments

    Args:
        releases (list): List of chart releases, charts and values files
            Example: "release1 --chart chart_name --values file"
    Returns:
        dict: Dictionary of packages sys.argv[-1and build-args
            Example: { "release1": { "chart": "chart_name", "values": "file", "chart-folder": "file" } }
    """
    parsed_args = {}
    current_arg = None
    for rel in releases:
        if rel.startswith("--chart"):
            chart = rel.split("=")[1]
            parsed_args[current_arg]["chart"] = chart
        elif rel.startswith("--values"):
            values = rel.split("=")[1]
            parsed_args[current_arg]["values"] = values
        elif rel.startswith("--folder"):
            chart_folder = rel.split("=")[1]
            parsed_args[current_arg]["chart-folder"] = chart_folder
        else:
            current_arg = rel
            parsed_args[rel] = {
                "chart": rel,
                "values": f"{rel}-helm-values",
                "chart-folder": rel,
            }
    return parsed_args


if __name__ == "__main__":
    command = sys.argv[1]
    stage = sys.argv[2]
    dry_run = sys.argv[-1] == "--dry-run"
    if dry_run:
        print("Running with --dry-run")
        sys.argv.pop(-1)
    packages = sys.argv[3:]
    print(packages)
    if command == "update-versions":
        if sys.argv[-1].startswith("--version"):
            version = sys.argv[-1].split("=")[1]
            sys.argv.pop(-1)
        else:
            version = "prerelease"
        update_versions(stage, packages, [], version, dry_run)
    if command == "build":
        packages = parse_build_command(packages)
        build(stage, packages, dry_run)
    if command == "deploy":
        packages = parse_deploy_command(packages)
        print("Deploying", packages)
        deploy(stage, packages, dry_run)
