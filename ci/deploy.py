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


def _replace_underscore(package: str):
    return package.replace("_", "-")


async def helm_upgrade(
    client: dagger.Client,
    k8s: dagger.Container,
    release_name: str,
    chart_name: str,
    values_file: str,
    dry_run: bool,
):
    helm_command = [
        "helm",
        "upgrade",
        "-i",
        "-f",
        "values.yaml",
        release_name,
        f"pipeline/{chart_name}",
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
                values_file,
            )
        )
        .with_exec(helm_command)
    )


async def deploy_package(packages: dict, cluster_name: str, cluster_alias: str, dry_run: bool):
    async with dagger.Connection() as client:
        # list of bash command to be executed in the dager container
        # with the purpose of conecting to the cluster
        configure_aws_eks_command_list = [
            "sh",
            "-c",
            f"aws eks update-kubeconfig --region us-east-1 --name {cluster_name} --alias {cluster_alias}",
        ]

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
                configure_aws_eks_command_list
            )
        )
        async with anyio.create_task_group() as tg:
            for package in packages:
                tg.start_soon(
                    helm_upgrade,
                    client,
                    k8s,
                    package,
                    packages[package]["chart"],
                    packages[package]["values"],
                    dry_run,
                )
    print("Deployed packages successfully")


def deploy_staging(packages: dict, dry_run: bool):
    anyio.run(deploy_package, packages, "staging", "staging", dry_run)

def deploy_production(packages: dict, dry_run: bool):
    anyio.run(deploy_package, packages, "staging", "staging", dry_run)