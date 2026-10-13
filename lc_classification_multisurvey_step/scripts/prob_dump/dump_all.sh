#!/usr/bin/env bash
# Dumps all 16 probability partitions (classifier_version 210 by default), N at a time.
# Usage: bash dump_all.sh <readonly_credentials.json> [parallel=4] [version=210]
# Output: ~/dumps/prob_part_<N>_v<version>.csv.gz and ~/dumps/part_<N>.log per partition.
set -uo pipefail  # keep going if one partition fails; its log says so
CRED="$1"  # credentials path, never printed
PAR="${2:-4}"  # partitions in flight
VER="${3:-210}"  # classifier_version
HERE="$(cd "$(dirname "$0")" && pwd)"  # folder of this script
PY="$HERE/.venv/bin/python"  # venv interpreter from setup.sh
mkdir -p "$HOME/dumps"  # output folder
echo "start: $(date -u +%FT%TZ)"  # wall clock start
seq 0 15 | xargs -P "$PAR" -I{} sh -c "$PY $HERE/prob_dump_timing.py $CRED {} $VER > $HOME/dumps/part_{}.log 2>&1"  # 16 partitions, PAR at a time
echo "end: $(date -u +%FT%TZ)"  # wall clock end
echo "parts_done: $(grep -l '^x16_estimate' $HOME/dumps/part_*.log | wc -l) of 16"  # logs that reached the last line
echo "rows_total: $(grep -h '^rows:' $HOME/dumps/part_*.log | awk '{s+=$2} END {print s}')"  # sum of rows
echo "gz_total_mb: $(du -cm $HOME/dumps/prob_part_*_v$VER.csv.gz | tail -1 | cut -f1)"  # total compressed size
