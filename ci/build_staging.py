import anyio
from utils import build, update_version, git_push


async def _build_package(packages: dict, dry_run: bool):
    async with anyio.create_task_group() as tg:
        for pkg in packages:
            z = zip(packages[pkg]["build-arg"], packages[pkg]["value"])
            build_args = []
            for arg, value in z:
                build_args.append((arg, value))
            tg.start_soon(build, pkg, build_args, dry_run)


async def _update_package_version(packages: list, dry_run: bool):
    async with anyio.create_task_group() as tg:
        for package in packages:
            tg.start_soon(update_version, package, "prerelease", dry_run)


def update_packages(packages, libs, dry_run: bool):
    anyio.run(_update_package_version, packages + libs, dry_run)
    anyio.run(git_push, dry_run)


def build_staging(packages: dict, dry_run: bool):
    anyio.run(_build_package, packages, dry_run)
