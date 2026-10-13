#!/usr/bin/env bash
# Creates .venv next to this script and installs the dump requirements.
# Usage: bash setup.sh   (re-runnable; reuses the venv if it exists)
set -euo pipefail  # stop on first error
HERE="$(cd "$(dirname "$0")" && pwd)"  # folder of this script
VENV="$HERE/.venv"  # venv location
[ -d "$VENV" ] || python3 -m venv "$VENV"  # create once
"$VENV/bin/pip" install -q --upgrade pip  # recent pip
"$VENV/bin/pip" install -q -r "$HERE/requirements.txt"  # psycopg2
"$VENV/bin/python" -c 'import psycopg2, pyarrow; print("psycopg2:", psycopg2.__version__, "pyarrow:", pyarrow.__version__)'  # proof it imports
echo "venv: $VENV"  # where it is
echo "run: $VENV/bin/python $HERE/prob_dump_timing.py <readonly_credentials.json> 0"  # the command
