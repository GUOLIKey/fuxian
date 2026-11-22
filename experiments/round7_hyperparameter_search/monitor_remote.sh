#!/usr/bin/env bash
set -u
dir=/root/autodl-tmp/fedvade_project/reproduction_runs/round7_hyperparameter_search
printf 'FedVaDE Hyperparameter Search\n\n'
if [ ! -f "$dir/progress.txt" ]; then
  printf 'Current Config: -- / 16\nStage: Not started\nEpoch: --\n\nLatest ARI: --\nLatest Purity: --\n\nStatus: UNKNOWN\n'
  exit 0
fi
IFS='|' read -r stage position total config path recorded < "$dir/progress.txt"
if [ "$stage" = coarse ]; then
  stage_name='Coarse Search'
  budget=150
else
  stage_name='Fine Search'
  budget=300
fi
epoch=0
ari='--'
purity='--'
if [ -f "$path/rounds.jsonl" ]; then
  line=$(tail -n 1 "$path/rounds.jsonl")
  epoch=$(printf '%s\n' "$line" | sed -n 's/.*"round": \([0-9][0-9]*\).*/\1/p')
  ari=$(printf '%s\n' "$line" | sed -n 's/.*"ari": \([-0-9.][0-9.eE+-]*\).*/\1/p')
  purity=$(printf '%s\n' "$line" | sed -n 's/.*"purity": \([-0-9.][0-9.eE+-]*\).*/\1/p')
  [ -n "$epoch" ] || epoch=0
  [ -n "$ari" ] || ari='--'
  [ -n "$purity" ] || purity='--'
fi
status=UNKNOWN
if [ "$recorded" = FINISHED ]; then
  status=FINISHED
elif [ "$recorded" = FAILED ] || [ -f "$dir/error.txt" ]; then
  status=FAILED
elif [ -f "$dir/search.pid" ] && kill -0 "$(cat "$dir/search.pid")" 2>/dev/null; then
  status=RUNNING
fi
printf 'Current Config: %s / %s\nStage: %s\nEpoch: %s / %s\n\nLatest ARI: %s\nLatest Purity: %s\n\nStatus: %s\n' \
  "$position" "$total" "$stage_name" "$epoch" "$budget" "$ari" "$purity" "$status"
if [ "$status" = FAILED ] && [ -f "$dir/error.txt" ]; then
  printf '\nError: '
  tail -n 1 "$dir/error.txt"
fi
