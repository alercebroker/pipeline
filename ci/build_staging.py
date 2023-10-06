import anyio
from utils import build, get_tags, update_version, git_push, update_chart


def _build_package(package_dir: str, dry_run: bool):
    tags = anyio.run(get_tags, package_dir)
    anyio.run(build, package_dir, tags, dry_run)


def _update_package_version(packages: list, dry_run: bool):
    for package in packages:
        anyio.run(update_version, package, "prerelease", dry_run)


def _update_chart_version(packages: list, dry_run: bool):
    for package in packages:
        anyio.run(update_chart, package, dry_run)


def build_staging(dry_run: bool):
    packages = [
        "correction_step",
        "feature_step",
        "lc_classification_step",
        "lightcurve-step",
        "magstats_step",
        "prv_candidates_step",
        "s3_step",
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
    _update_package_version(packages + libs, dry_run)
    _update_chart_version(packages, dry_run)
    anyio.run(git_push, dry_run)
    for package in packages:
        _build_package(package, dry_run)
