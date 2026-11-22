#!/usr/bin/env bash
set -u
dir=/root/autodl-tmp/fedvade_project/reproduction_runs/round8_checkpoint_finetune
printf 'FedVaDE Round 8 Monitor\n\n'
if [ ! -f "$dir/progress.json" ]; then
  printf 'Stage: UNKNOWN\nStatus: UNKNOWN\n'
  exit 0
fi
value() {
  sed -n "s/^[[:space:]]*\"$1\": \(.*\),$/\1/p" "$dir/progress.json" | head -1 | sed 's/^"//;s/"$//;s/^null$/--/'
}
stage=$(value stage)
group=$(value current_group)
epoch=$(value epoch)
budget=$(value budget)
ari=$(value latest_ari)
purity=$(value latest_purity)
best=$(value best_checkpoint)
best_ari=$(value best_ari_so_far)
[ -n "$best_ari" ] || best_ari="$ari"
current=$(value checkpoint_current)
total=$(value checkpoint_total)
status=UNKNOWN
if [ -f "$dir/completed.marker" ] && [ -s "$dir/task_exit_status.json" ] \
   && grep -q '"exit_code":0' "$dir/task_exit_status.json" \
   && [ -s "$dir/ROUND8_REPORT.md" ] && [ -s "$dir/finetune_metrics.csv" ] \
   && [ -s "$dir/inference_repeats.csv" ] && [ -s "$dir/best_finetuned_checkpoint/server.pt" ]; then
  status=FINISHED
elif [ -f "$dir/failed.marker" ] || [ "$stage" = FAILED ]; then
  status=FAILED
else
  now=$(date +%s)
  modified=$(stat -c %Y "$dir/progress.json")
  if [ $((now-modified)) -gt 120 ]; then
    status=STALE
  elif [ -f "$dir/task.pid" ] && kill -0 "$(cat "$dir/task.pid")" 2>/dev/null; then
    status=RUNNING
  fi
fi
printf 'Stage: %s\n\n' "$stage"
case "$stage" in
  CHECKPOINT_SCAN|CHECKPOINT_EVALUATION)
    printf 'Checkpoint: %s / %s\n' "$current" "$total" ;;
  RESUME_VALIDATION|SMOKE_TEST)
    printf 'Checkpoint: validation in progress or complete\n' ;;
  FINETUNE|FINAL_EVALUATION|FINISHED)
    printf 'Current Group: %s\nEpoch: %s / %s\n' "$group" "$epoch" "$budget" ;;
esac
printf '\nLatest ARI: %s\nLatest Purity: %s\n\nBest ARI So Far: %s\nBest Checkpoint: %s\nStatus: %s\n' "$ari" "$purity" "$best_ari" "$best" "$status"
if [ "$status" = FAILED ] && [ -f "$dir/error.txt" ]; then
  printf 'Error: '
  tail -n 1 "$dir/error.txt"
fi
