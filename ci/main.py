import sys
from build_staging import build_staging, update_packages
from deploy_staging import deploy_staging


def update_versions(stage, packages, libs, dry_run):
    if stage == "staging":
        update_packages(packages, libs, dry_run)
    elif stage == "production":
        print("Production update not implemented yet")
    else:
        raise ValueError(
            f'Invalid stage "{stage}". Valid stages are: staging, production'
        )


def build(stage, packages, dry_run):
    if stage == "staging":
        build_staging(packages, dry_run)
    elif stage == "production":
        print("Production build not implemented yet")
    else:
        raise ValueError(
            f'Invalid stage "{stage}". Valid stages are: staging, production'
        )


def deploy(stage, dry_run):
    if stage == "staging":
        deploy_staging(dry_run)
    elif stage == "production":
        print("Production deploy not implemented yet")
    else:
        raise ValueError(
            f'Invalid stage "{stage}". Valid stages are: staging, production'
        )


if __name__ == "__main__":
    command = sys.argv[1]
    stage = sys.argv[2]
    packages = sys.argv[3]
    packages = packages.split(",")
    dry_run = False
    if len(sys.argv) > 4:
        dry_run = sys.argv[4]
        dry_run = dry_run == "--dry-run"
        print("Running with --dry-run")
    if command == "update-versions":
        update_versions(stage, packages, [], dry_run)
    if command == "build":
        build(stage, packages, dry_run)
    if command == "deploy":
        deploy(stage, dry_run)
