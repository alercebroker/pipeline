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


def build_staging(dry_run: bool):
    packages = [
        "correction_step",
        "feature_step",
        "lightcurve-step",
        "magstats_step",
        "prv_candidates_step",
        "scribe",
        "sorting_hat_step",
        "xmatch_step",
    ]
    libs = [
        "libs/apf",
        "libs/db-plugins",
        "libs/lc_classifier",
        "libs/survey_parser_plugins",
        "libs/turbo-fats",
    ]
    anyio.run(_update_package_version, packages + libs, dry_run)
    anyio.run(_update_chart_version, packages, dry_run)
    anyio.run(git_push, dry_run)
    anyio.run(_build_package, packages, dry_run)
