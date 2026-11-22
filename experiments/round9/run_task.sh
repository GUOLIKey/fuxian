#!/usr/bin/env bash
set -u
BASE=/root/autodl-tmp/fedvade_project/reproduction_runs/round9_fedvade_optimization
PYTHON=/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python
cd "$BASE" || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf '%s\n' "started_at=$(date -Iseconds)" "command=$PYTHON $BASE/task_round9.py" > launch_info.txt
"$PYTHON" "$BASE/task_round9.py" >> "$BASE/task.log" 2>&1
code=$?
printf '%s\n' "wrapper_exit_code=$code" "finished_at=$(date -Iseconds)" >> launch_info.txt
exit "$code"
