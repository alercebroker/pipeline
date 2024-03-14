import sys
import dagger
import os
import pathlib
import sys
from enum import Enum


def _get_publish_secret(client):
    gtoken = os.environ["GHCR_TOKEN"]
    secret = client.set_secret("gtoken", gtoken)
    return secret


async def _publish_container(
    container: dagger.Container,
    output_package_name,
    tags: list,
    secret: dagger.Secret,
):
    for tag in tags:
        addr = await container.with_registry_auth(
            f"ghcr.io/alercebroker/{output_package_name}:{tag}",
            "alerceadmin",
            secret,
        ).publish(f"ghcr.io/alercebroker/{output_package_name}:{tag}")
        print(f"published image at: {addr}")


async def git_push(dry_run: bool):
    config = dagger.Config(log_output=sys.stdout)

    async with dagger.Connection(config) as client:
        path = pathlib.Path().cwd().parent.absolute()
        container = (
            client.container()
            .from_("alpine:latest")
            .with_exec(["apk", "add", "--no-cache", "git"])
            .with_exec(
                ["git", "config", "--global", "user.name", '"@alerceadmin"']
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
            .with_directory(
                "/pipeline",
                client.host().directory(
                    str(path),
                    exclude=[".venv/", "**/.venv/", ".terraform/",
                             "**/.terraform", "tests/", "**/tests/",]
                ),
            )
            .with_workdir("/pipeline")
            .with_exec(["git", "status"])
        )
        if not dry_run:
            await (
                container.with_exec(["git", "add", "--ignore-removal", "."])
                .with_exec(["git", "commit", "-m", "chore: update version"])
                .with_exec(["git", "push"])
            )
        else:
            await container


async def update_version(
    source: dagger.Container,
    path: pathlib.Path,
    package_dir: str,
    version: str,
    also_update_chart: bool,
    dry_run: bool,
):
    update_version_command = ["poetry", "version", version]
    if dry_run:
        update_version_command.append("--dry-run")
    source = (
        source.with_workdir(f"/pipeline/{package_dir}")
        .with_exec(update_version_command)
        .with_exec(["poetry", "version", "--short"])
    )
    await source.directory(f"/pipeline/{package_dir}").export(
        str(path / package_dir)
    )
    if also_update_chart:
        new_version = await source.stdout()
        new_version = new_version.strip()
        updated = update_chart(source, package_dir, new_version, dry_run)
        await updated.directory(f"/pipeline/charts/{package_dir}").export(
            str(path / f"charts/{package_dir}")
        )


async def build(
    client: dagger.Client,
    package_dir: str,
    build_args: list,
    output_package_name,
    dry_run: bool,
):
    path = pathlib.Path().cwd().parent.absolute()
    # get build context directory
    context_dir = client.host().directory(
        str(path),
        exclude=[".venv/", "**/.venv/", ".terraform/",
                 "**/.terraform", "tests/", "**/tests/"]
    )
    # build using Dockerfile
    bargs = []
    for build_arg in build_args:
        barg = dagger.BuildArg(build_arg[0], build_arg[1])
        bargs.append(barg)
    print(f"Building image with build-args: {bargs}")
    image_ref = await client.container().build(
        context=context_dir,
        dockerfile=f"{package_dir}/Dockerfile",
        build_args=bargs,
    )
    tags = await get_tags(client, package_dir)
    print(f"Built image with tag: {tags}")

    if not dry_run:
        for tag in tags:
            print(
                f"Publishing image to ghcr.io/alercebroker/{output_package_name}:{tag}"
            )
        # publish the resulting container to a registry
        secret = _get_publish_secret(client)
        await _publish_container(image_ref, output_package_name, tags, secret)
    else:
        for tag in tags:
            print(
                f"Running with dry run: Skipping publish to ghcr.io/alercebroker/{output_package_name}:{tag}"
            )


async def publish_lib(client: dagger.Client, package_dir, dry_run: bool):
    path = pathlib.Path().cwd().parent.absolute()
    command = ["poetry", "publish", "--build"]
    if dry_run:
        command.append("--dry-run")
    # get build context directory
    source = (
        client.container()
        .from_("python:3.11-slim")
        .with_exec(["pip", "install", "poetry"])
        .with_directory(
            f"/pipeline/{package_dir}",
            client.host().directory(
                str(path / package_dir),
                exclude=[".venv/", "**/.venv/", ".terraform/",
                         "**/.terraform", "tests/", "**/tests/"]
            ),
        )
        .with_workdir(f"/pipeline/{package_dir}")
        .with_exec(
            ["poetry", "config", "pypi-token.pypi", os.environ["PYPI_TOKEN"]]
        )
        .with_exec(command)
    )
    out = await source.stdout()
    print(out)


async def get_tags(client: dagger.Client, package_dir: str) -> list:
    path = pathlib.Path().cwd().parent.absolute()
    # get build context directory
    source = (
        client.container()
        .from_("python:3.11-slim")
        .with_exec(["pip", "install", "poetry"])
        .with_directory(
            "/pipeline",
            client.host().directory(
                str(path),
                exclude=[".venv/", "**/.venv/", ".terraform/",
                         "**/.terraform", "tests/", "**/tests/",]
            ),
        )
    )
    runner = source.with_workdir(f"/pipeline/{package_dir}").with_exec(
        ["poetry", "version", "--short"]
    )
    out = await runner.stdout()
    return ["rc", out.strip("\n")]


def update_chart(
    container: dagger.Container, chart_name, app_version, dry_run: bool
):
    script = [
        "poetry",
        "run",
        "python",
        "chart_script.py",
        chart_name,
        app_version,
    ]
    if dry_run:
        script.append("--dry-run")
    return container.with_workdir("/pipeline/ci").with_exec(script)


class Stage(str, Enum):
    staging = "staging"
    production = "production"
