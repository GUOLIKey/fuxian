#!/usr/bin/env bash
BASE=/root/autodl-tmp/fedvade_project/reproduction_runs/round9_fedvade_optimization
FILE="$BASE/progress.json"
field() {
  local key="$1" value
  value=$(sed -n "s/^[[:space:]]*\"$key\":[[:space:]]*//p" "$FILE" 2>/dev/null | head -n 1)
  value=${value%,}
  value=${value#\"}
  value=${value%\"}
  if [[ -z "$value" || "$value" == null ]]; then printf -- '-'; else printf '%s' "$value"; fi
}
if [[ ! -f "$FILE" ]]; then
  printf 'Stage: -\nCurrent Config: -\nEpoch: -\nLatest ARI: -\nLatest Purity: -\nBest ARI So Far: -\nBest Checkpoint: -\nGitHub: PENDING\nStatus: UNKNOWN\n'
  exit 0
fi
stage=$(field stage)
config=$(field current_config)
current=$(field config_current)
total=$(field config_total)
epoch=$(field epoch)
epoch_total=$(field epoch_total)
ari=$(field latest_ari)
purity=$(field latest_purity)
best=$(field best_ari_so_far)
checkpoint=$(field best_checkpoint)
github=$(field github)
status=$(field status)
if [[ -f "$BASE/github_publish_status.json" ]]; then
  published=$(sed -n 's/^[[:space:]]*"status":[[:space:]]*"\([^"]*\)".*/\1/p' "$BASE/github_publish_status.json" | head -n 1)
  [[ -n "$published" ]] && github=$published
fi
printf 'Stage: %s\nCurrent Config: %s / %s (%s)\nEpoch: %s / %s\n\nLatest ARI: %s\nLatest Purity: %s\n\nBest ARI So Far: %s\nBest Checkpoint: %s\n\nGitHub: %s\nStatus: %s\n' \
  "$stage" "$current" "$total" "$config" "$epoch" "$epoch_total" \
  "$ari" "$purity" "$best" "$checkpoint" "$github" "$status"
