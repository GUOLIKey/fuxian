#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/fedvade_project
PY="$ROOT/.venv/bin/python"
SCRIPTS="$ROOT/experiments/multica_factorial_v1/scripts"
BASE="$ROOT/experiments/multica_factorial_v1"
RUNS="$ROOT/reproduction_runs/multica_factorial_v1"
STAGE="$RUNS/stage_a"
mkdir -p "$STAGE" "$RUNS/logs" "$BASE/results" "$BASE/diagnostics"

export PYTHONUNBUFFERED=1
"$PY" "$SCRIPTS/make_manifest.py" || true

check_disk() {
  local avail
  avail=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
  if [ "${avail:-0}" -lt 1500000 ]; then
    echo "DISK_LOW: ${avail} KiB available" >&2
    return 1
  fi
}

for seed in 0 1 2; do
  if [ ! -f "$RUNS/pretrain/seed${seed}/pretrain.pt" ]; then
    echo "MISSING_PRETRAIN seed${seed}" >&2
    exit 1
  fi
done

for group in A B C D; do
  for seed in 0 1 2; do
    out="$STAGE/${group}_seed${seed}"
    log="$RUNS/logs/${group}_seed${seed}.log"
    err="$RUNS/logs/${group}_seed${seed}.err.log"
    if [ -f "$out/result.json" ] && grep -q '"status": "ok"' "$out/result.json"; then
      echo "SKIP_OK ${group}_seed${seed}"
      continue
    fi
    if [ -e "$out" ]; then
      echo "REFUSE_EXISTING_INCOMPLETE ${out}" >&2
      "$PY" "$SCRIPTS/make_manifest.py" || true
      exit 1
    fi
    check_disk
    echo "START ${group}_seed${seed} $(date -u +%FT%TZ)"
    "$PY" "$SCRIPTS/run_factorial.py" \
      --group "$group" --seed "$seed" --rounds 300 --save-every 25 \
      --device cuda --pretrain "$RUNS/pretrain/seed${seed}/pretrain.pt" \
      --out "$out" >"$log" 2>"$err"
    echo "DONE ${group}_seed${seed} $(date -u +%FT%TZ)"
    "$PY" "$SCRIPTS/make_manifest.py" || true
  done
done

"$PY" "$SCRIPTS/make_manifest.py" || true
date -u +%FT%TZ > "$RUNS/stage_a_done.txt"
echo ALL_STAGE_A_DONE
