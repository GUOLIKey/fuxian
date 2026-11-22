#!/usr/bin/env bash
set -u
BASE=/root/autodl-tmp/fedvade_project/reproduction_runs/round10_non_iid_generation
PYTHON=/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python
cd "$BASE" || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'started_at=%s\ncommand=%s\n' "$(date -Iseconds)" "$PYTHON $BASE/task_round10.py" > launch_info.txt
"$PYTHON" "$BASE/task_round10.py" >> task.log 2>&1
code=$?
printf 'wrapper_exit_code=%s\nfinished_at=%s\n' "$code" "$(date -Iseconds)" >> launch_info.txt
exit "$code"
