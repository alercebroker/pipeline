import anyio
import dagger
import sys
import pathlib
from utils import build, update_version, git_push, publish_lib


def is_library(package: str):
    libs = [
        "lc_classifier",
        "mhps",
        "P4J",
        "alerce_classifiers",
        "libs/db-plugins",
        "apf",
        "survey_parser_plugins",
        "test_utils",
    ]
    return package in libs


async def _build_package(packages: dict, dry_run: bool):
    config = dagger.Config(log_output=sys.stdout)

    async with dagger.Connection(config) as client:
        async with anyio.create_task_group() as tg:
            for pkg in packages:
                z = zip(packages[pkg]["build-arg"], packages[pkg]["value"])
                build_args = []
                for arg, value in z:
                    build_args.append((arg, value))
                if not is_library(pkg):
                    tg.start_soon(
                        build,
                        client,
                        packages[pkg]["package-dir"],
                        build_args,
                        pkg,
                        dry_run,
                    )
                else:
                    tg.start_soon(
                        publish_lib,
                        client,
                        packages[pkg]["package-dir"],
                        dry_run,
                    )


async def _update_package_version(packages: list, version: str, dry_run: bool):
    config = dagger.Config(log_output=sys.stdout)
    path = pathlib.Path().cwd().parent.absolute()
    async with dagger.Connection(config) as client:
        source = (
            client.container()
            .from_("python:3.11-slim")
            .with_exec(["apt", "update"])
            .with_exec(["apt", "install", "git", "-y"])
            .with_exec(
                [
                    "git",
                    "config",
                    "--global",
                    "user.name",
                    '"@alerceadmin"',
                ]
            )
            .with_exec(
                [
                    "git",
                    "config",
                    "--global",
                    "user.email",
                    "alerceadmin@users.noreply.github.com",
                ]
            )
            .with_exec(["pip", "install", "poetry"])
            .with_directory(
                "/pipeline",
                client.host().directory(
                    str(path),
                    exclude=[".venv/", "**/.venv/", "*/.venv/",
                             "*.venv", "tests/", "**/tests/"],
                ),
            )
        )
        async with anyio.create_task_group() as tg:
            for package in packages:
                also_update_chart = not is_library(package)
                tg.start_soon(
                    update_version,
                    source,
                    path,
                    package,
                    version,
                    also_update_chart,
                    dry_run,
                )


def update_packages(packages, version, dry_run: bool):
    anyio.run(_update_package_version, packages, version, dry_run)
    anyio.run(git_push, dry_run)


def build_packages(packages: dict, dry_run: bool):
    anyio.run(_build_package, packages, dry_run)
