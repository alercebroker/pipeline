import sys
import dagger
import os
import pathlib
import sys


def _get_publish_secret(client):
    gtoken = os.environ["GHCR_TOKEN"]
    secret = client.set_secret("gtoken", gtoken)
    return secret


async def _publish_container(
    container: dagger.Container,
    package_name: str,
    tags: list,
    secret: dagger.Secret,
):
    for tag in tags:
        addr = await container.with_registry_auth(
            f"ghcr.io/alercebroker/{package_name}:{tag}",
            "alerceadmin",
            secret,
        ).publish(f"ghcr.io/alercebroker/{package_name}:{tag}")
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
                    str(path), exclude=[".venv/", "**/.venv/"]
                ),
            )
            .with_workdir("/pipeline")
            .with_exec(["git", "status"])
        )
        if not dry_run:
            await (
                container.with_exec(["git", "add", "."])
                .with_exec(["git", "commit", "-m", "chore: update version"])
                .with_exec(["git", "push"])
            )
        else:
            await container


async def update_version(package_dir: str, version: str, dry_run: bool):
    config = dagger.Config(log_output=sys.stdout)

    async with dagger.Connection(config) as client:
        path = pathlib.Path().cwd().parent.absolute()
        # get build context directory
        source = (
            client.container()
            .from_("python:3.11-slim")
            .with_exec(["apt", "update"])
            .with_exec(["apt", "install", "git", "-y"])
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
            .with_exec(["pip", "install", "poetry"])
            .with_directory(
                "/pipeline",
                client.host().directory(
                    str(path), exclude=[".venv/", "**/.venv/"]
                ),
            )
        )
        update_version_command = ["poetry", "version", version]
        if dry_run:
            update_version_command.append("--dry-run")

        new_version = await (
            source.with_workdir(f"/pipeline/{package_dir}")
            .with_exec(update_version_command)
            .with_exec(["poetry", "version", "--short"])
            .stdout()
        )
        new_version = new_version.strip()
        await update_chart(source, path, package_dir, new_version, dry_run)
        await source.directory(f"/pipeline/{package_dir}").export(
            str(path / package_dir)
        )
        await source.directory("/pipeline/charts").export(str(path / "charts"))


async def build(package_dir: str, dry_run: bool):
    config = dagger.Config(log_output=sys.stdout)

    async with dagger.Connection(config) as client:
        path = pathlib.Path().cwd().parent.absolute()
        # get build context directory
        context_dir = client.host().directory(
            str(path), exclude=[".venv/", "**/.venv/"]
        )
        # build using Dockerfile
        image_ref = await client.container().build(
            context=context_dir, dockerfile=f"{package_dir}/Dockerfile"
        )
        tags = await get_tags(client, package_dir)
        print(f"Built image with tag: {tags}")

        if not dry_run:
            print("Publishing image")
            # publish the resulting container to a registry
            secret = _get_publish_secret(client)
            await _publish_container(image_ref, package_dir, tags, secret)


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
                str(path), exclude=[".venv/", "**/.venv/"]
            ),
        )
    )
    runner = source.with_workdir(f"/pipeline/{package_dir}").with_exec(
        ["poetry", "version", "--short"]
    )
    out = await runner.stdout()
    return ["rc", out.strip("\n")]


async def update_chart(
    container: dagger.Container, path, chart_name, app_version, dry_run: bool
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
    await container.with_workdir("/pipeline/ci").with_exec(script)
