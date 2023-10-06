import sys
from build_staging import build_staging
from deploy_staging import deploy_staging


def build(stage, dry_run):
    if stage == "staging":
        build_staging(dry_run)
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
    dry_run = False
    if len(sys.argv) > 3:
        dry_run = sys.argv[3]
        dry_run = dry_run == "--dry-run"
        print("Running with --dry-run")
    if command == "build":
        build(stage, dry_run)
    if command == "deploy":
        deploy(stage, dry_run)
