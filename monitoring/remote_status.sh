#!/usr/bin/env bash
# Read-only live panel for the frozen-client global ACD experiment.
set -u
ROOT=/root/autodl-tmp/fedvade_project
EXP="$ROOT/experiments/acd_f01_global_v1"
PY="$ROOT/.venv-cpu/bin/python"
INTERVAL=5
WATCH=0
[ "${1:-}" = "--watch" ] && WATCH=1
[ "${2:-}" != "" ] && INTERVAL="$2"

render() {
  printf '\033[2J\033[H'
  echo "F01 ACD global experiment | $(date -u +%FT%TZ)"
  echo
  "$PY" "$EXP/monitoring/panel.py" "$EXP"
  echo
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
      --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
  else
    echo "GPU: nvidia-smi unavailable"
  fi
  echo "CPU/RAM: $(nproc) vCPU | $(free -h | awk '/^Mem:/ {print $3 " used / " $2}')"
  echo "Disk: $(df -h /root/autodl-tmp | awk 'NR==2 {print $4 " free / " $2 " (" $5 " used)"}')"
  echo
  latest=$(ls -t "$EXP"/logs/*.log 2>/dev/null | head -1 || true)
  if [ -n "$latest" ]; then
    echo "Latest log: $(basename "$latest")"
    tail -c 1400 "$latest" 2>/dev/null | tr '\r' '\n' | grep -vE '^\s*$' | tail -n 4
  else
    echo "Latest log: none"
  fi
  echo
  echo "Recent errors:"
  found=0
  for f in "$EXP"/logs/*.log; do
    [ -f "$f" ] || continue
    if grep -qE "Traceback|RuntimeError|Error|FAILED" "$f" 2>/dev/null; then
      found=1
      echo "--- $(basename "$f")"
      grep -E "Traceback|RuntimeError|Error|FAILED" "$f" | tail -n 2
    fi
  done
  [ "$found" -eq 1 ] || echo "none"
  echo
  echo "Next refresh: ${INTERVAL}s | Ctrl-C to stop"
}

if [ "$WATCH" -eq 1 ]; then
  while true; do render; sleep "$INTERVAL"; done
else
  render
fi
