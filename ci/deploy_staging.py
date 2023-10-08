import dagger
import os
import sys
import anyio
import pathlib
from multiprocessing import Process, Pool


def env_variables(envs: dict[str, str]):
    def env_variables_inner(ctr: dagger.Container):
        for key, value in envs.items():
            ctr = ctr.with_env_variable(key, value)
        return ctr

    return env_variables_inner


def get_values(client: dagger.Client, path: str, ssm_parameter_name: str):
    def get_values_inner(ctr: dagger.Container):
        ctr = (
            ctr.with_directory(
                "/pipeline",
                client.host().directory(path, exclude=[".venv/", "**/.venv/"]),
            )
            .with_workdir("/pipeline/ci")
            .with_exec(["python", "-m", "pip", "install", "poetry"])
            .with_exec(["poetry", "install"])
            .with_exec(
                [
                    "poetry",
                    "run",
                    "python",
                    "ssm.py",
                    ssm_parameter_name,
                ]
            )
        )
        return ctr

    return get_values_inner


async def helm_upgrade(package: str, dry_run: bool):
    config = dagger.Config(log_output=sys.stdout)

    async with dagger.Connection(config) as client:
        k8s = (
            client.container()
            .from_("alpine/k8s:1.27.5")
            .with_(
                env_variables(
                    {
                        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                        "AWS_SECRET_ACCESS_KEY": os.environ[
                            "AWS_SECRET_ACCESS_KEY"
                        ],
                        "AWS_SESSION_TOKEN": os.environ["AWS_SESSION_TOKEN"],
                    },
                )
            )
            .with_env_variable(
                "AWS_DEFAULT_REGION", os.environ["AWS_DEFAULT_REGION"]
            )
            .with_exec(
                [
                    "sh",
                    "-c",
                    """
                    aws eks update-kubeconfig \
                    --region us-east-1 \
                    --name staging \
                    --alias staging
                    """,
                ]
            )
        )
        helm_command = [
            "helm",
            "upgrade",
            "-i",
            "-f",
            "values.yaml",
            package,
            f"pipeline/{package}",
        ]
        if dry_run:
            helm_command.append("--dry-run")

        await (
            k8s.with_exec(["kubectl", "config", "get-contexts"])
            .with_exec(
                [
                    "helm",
                    "repo",
                    "add",
                    "pipeline",
                    "https://alercebroker.github.io/pipeline",
                ]
            )
            .with_(
                get_values(
                    client,
                    str(pathlib.Path().cwd().parent.absolute()),
                    f"{package}-helm-values",
                )
            )
            .with_exec(helm_command)
        )


async def deploy_package(packages: list, dry_run: bool):
    for package in packages:
        async with anyio.create_task_group() as tg:
            tg.start_soon(helm_upgrade, package, dry_run)


def deploy_staging(packages: list, dry_run: bool):
    anyio.run(deploy_package, packages, dry_run)
