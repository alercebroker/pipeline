import anyio
from utils import build, update_version, git_push, update_chart


async def _build_package(packages: list, dry_run: bool):
    async with anyio.create_task_group() as tg:
        for package in packages:
            tg.start_soon(build, package, dry_run)


async def _update_package_version(packages: list, dry_run: bool):
    async with anyio.create_task_group() as tg:
        for package in packages:
            tg.start_soon(update_version, package, "prerelease", dry_run)


async def _update_chart_version(packages: list, dry_run: bool):
    async with anyio.create_task_group() as tg:
        for package in packages:
            tg.start_soon(update_chart, package, dry_run)


def update_packages(packages, libs, dry_run: bool):
    anyio.run(_update_package_version, packages + libs, dry_run)
    anyio.run(_update_chart_version, packages, dry_run)
    anyio.run(git_push, dry_run)


def build_staging(packages: list, dry_run: bool):
    anyio.run(_build_package, packages, dry_run)
