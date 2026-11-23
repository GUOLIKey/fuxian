#!/usr/bin/env bash
# Launch the 12 paired warm-start runs. Training survives SSH/monitor disconnects.
set -euo pipefail
ROOT=/root/autodl-tmp/fedvade_project
EXP="$ROOT/experiments/multica_pretrained_2x2_v1"
PY="$ROOT/.venv-cpu/bin/python"
mkdir -p "$EXP/logs" "$EXP/runs"
: > "$EXP/logs/pids.txt"
for model in AUTHOR F01; do
  for seed in 0 1 2; do
    for agg in off on; do
      if [ "$agg" = off ]; then init="$EXP/init/${model}_seed${seed}_base.pt";
      else init="$EXP/init/${model}_seed${seed}_on.pt"; fi
      out="$EXP/runs/${model}_seed${seed}_${agg}"
      log="$EXP/logs/${model}_seed${seed}_${agg}.log"
      rm -rf "$out"
      mkdir -p "$out"
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup "$PY" \
        "$EXP/scripts/train_runner.py" --init "$init" --aggregation "$agg" \
        --out "$out" --rounds 300 --save-every 25 > "$log" 2>&1 &
      echo "$! ${model} ${seed} ${agg}" >> "$EXP/logs/pids.txt"
    done
  done
done
sleep 2
cat "$EXP/logs/pids.txt"
