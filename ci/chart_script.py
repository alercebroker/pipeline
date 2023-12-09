import re
import sys


def update_app_version(package: str, new_version: str, dry_run: bool):
    """Updates the appVersion in the Chart.yaml file

    Parameters:
        package (str): The name of the package. Also the directory of the chart.
        new_version (str): The version to update to
            It should match the version in the pyproject.toml file of the same package
        dry_run (bool): If True, it will not write the changes to the file
    """
    with open(f"../charts/{package}/Chart.yaml", "r") as f:
        original = f.read()
        f.seek(0)
        match = re.search(r"appVersion: (\d+).(\d+).(\d+).*", original)
    with open(f"../charts/{package}/Chart.yaml", "w") as f:
        if not match:
            chart = re.sub(
                r"appVersion: \"(\d+).(\d+).(\d+).*\"",
                f"appVersion: {new_version}",
                original,
            )
        else:
            chart = re.sub(
                r"appVersion: (\d+).(\d+).(\d+).*",
                f"appVersion: {new_version}",
                original,
            )
        print(f"Updating {package} appVersion to {new_version}")
        if not dry_run:
            f.write(chart)


def update_chart_version(package: str, version: str, dry_run: bool = False):
    """Updates the chart version in the Chart.yaml file

    If no version is provided, it will increment the patch version

    Parameters:
        package (str): The name of the package. Also the directory of the chart.
        version (str): The version to update to
            It must follow SemVer 2.0. See https://semver.org/
            The version can't include modifier suffixes like "a" or "b"
        dry_run (bool): If True, it will not write the changes to the file
    """
    with open(f"../charts/{package}/Chart.yaml", "r") as f:
        original = f.read()
        f.seek(0)
        match = re.search(
            r"version: (?P<version>(\d+).(\d+).(\d+).*)", original
        )
    with open(f"../charts/{package}/Chart.yaml", "w") as f:
        if not match:
            raise ValueError("Could not find valid version in Chart.yaml")
        chart = re.sub(
            r"version: (\d+).(\d+).(\d+).*",
            f"version: {version}",
            original,
        )
        print(f"Updating chart {package} version to {version}")
        if not dry_run:
            f.write(chart)


def get_package_version(package: str):
    with open(f"../{package}/pyproject.toml", "r") as f:
        pyproject = f.read()
        poetry_section = re.search(
            r"\[tool\.poetry\]([\s\S]*?)\n\[[^\[\]]+\]|\[tool\.poetry\]([\s\S]*?)$",
            pyproject,
        )
        if not poetry_section:
            raise ValueError("Could not find poetry section in pyproject.toml")
        section = poetry_section.group(1)
        match = re.search(
            r'version = "(?P<version>(\d+).(\d+).(\d+).*)"', section
        )
        if not match:
            raise ValueError("Could not find version in pyproject.toml")
        version = match.group("version")
        return version


def main(package: str, package_version, dry_run: bool):
    update_app_version(package, package_version, dry_run)
    if "a" in package_version:
        modifier = package_version[package_version.index("a") + 1 :]
        package_version = package_version[: package_version.index("a")]
        package_version = f"{package_version}-{modifier}"
    update_chart_version(package, package_version, dry_run=dry_run)


if __name__ == "__main__":
    package = sys.argv[1]
    package_version = sys.argv[2]
    dry_run = False
    if len(sys.argv) > 3:
        dry_run = sys.argv[3] == "--dry-run"
    main(package, package_version, dry_run)
