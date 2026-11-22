#!/usr/bin/env bash
BASE=/root/autodl-tmp/fedvade_project/reproduction_runs/round10_non_iid_generation
FILE="$BASE/progress.json"
field() {
  local key="$1" value
  value=$(sed -n "s/^[[:space:]]*\"$key\":[[:space:]]*//p" "$FILE" 2>/dev/null | head -n 1)
  value=${value%,}; value=${value#\"}; value=${value%\"}
  if [[ -z "$value" || "$value" == null ]]; then printf -- '--'; else printf '%s' "$value"; fi
}
if [[ ! -f "$FILE" ]]; then
  printf 'Stage: --\nCurrent Dataset: --\nCurrent Scenario: --\nCurrent Run: --\nEpoch: -- / --\nLatest ARI: --\nLatest Purity: --\nRare-mode Coverage: --\nGeneration MMD: --\nGitHub: PENDING\nStatus: UNKNOWN\n'
  exit 0
fi
stage=$(field stage); dataset=$(field current_dataset); scenario=$(field current_scenario)
run=$(field current_run); epoch=$(field epoch); budget=$(field epoch_total)
ari=$(field latest_ari); purity=$(field latest_purity)
coverage=$(field rare_mode_coverage); mmd=$(field generation_mmd)
github=$(field github); status=$(field status)
if [[ -f "$BASE/github_publish_status.json" ]]; then
  actual=$(sed -n 's/^[[:space:]]*"status":[[:space:]]*"\([^"]*\)".*/\1/p' "$BASE/github_publish_status.json" | head -n 1)
  [[ -n "$actual" ]] && github=$actual
fi
if [[ "$status" == FINISHED ]]; then
  if [[ ! -f "$BASE/task_exit_status.json" || ! -f "$BASE/ROUND10_REPORT.md" || ! -f "$BASE/noniid_baseline_metrics.csv" ]]; then
    status=UNKNOWN
  elif ! grep -q '"exit_code": 0' "$BASE/task_exit_status.json"; then
    status=FAILED
  fi
elif [[ "$status" == RUNNING ]]; then
  now=$(date +%s); modified=$(stat -c %Y "$FILE" 2>/dev/null || echo 0)
  if (( now - modified > 900 )); then status=STALE; fi
fi
printf 'Stage: %s\nCurrent Dataset: %s\nCurrent Scenario: %s\nCurrent Run: %s\nEpoch: %s / %s\n\nLatest ARI: %s\nLatest Purity: %s\n\nRare-mode Coverage: %s\nGeneration MMD: %s\n\nGitHub: %s\nStatus: %s\n' \
  "$stage" "$dataset" "$scenario" "$run" "$epoch" "$budget" "$ari" "$purity" "$coverage" "$mmd" "$github" "$status"
