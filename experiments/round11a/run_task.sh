#!/bin/bash
set -euo pipefail
BASE=/root/autodl-tmp/fedvade_project/reproduction_runs/round11a_frozen_diagnostics
PY=/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python
cd "$BASE"
if [ -e task_started.json ]; then
  echo "Round 11A already started; refusing duplicate launch"
  exit 1
fi
{
  echo "started_at=$(date -Iseconds)"
  echo "command=$PY $BASE/task_round11a.py"
} > launch_info.txt
nohup setsid "$PY" "$BASE/task_round11a.py" > "$BASE/task.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > task.pid
echo "pid=$pid" >> launch_info.txt
sleep 2
if kill -0 "$pid" 2>/dev/null; then
  echo "RUNNING pid=$pid"
  echo "launcher_exit_code=0" >> launch_info.txt
  exit 0
else
  echo "FAILED to start pid=$pid"
  echo "launcher_exit_code=1" >> launch_info.txt
  exit 1
fi
