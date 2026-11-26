#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=/root/autodl-tmp/fedvade_project/reproduction_runs/c13_dpgmm_3datasets_v1
case "${1:-original}" in
  original)
    CAMPAIGN="$BASE/formal_cv_selected_20261126"
    seeds=(0 1 2)
    groups=(A B C)
    ;;
  supplement)
    CAMPAIGN="$BASE/formal_ab_seeds34_20261127"
    seeds=(3 4)
    groups=(A B)
    ;;
  *)
    echo "Usage: $0 [original|supplement]" >&2
    exit 2
    ;;
esac
DATA=/root/autodl-tmp/fedvade_project/fuxian-round2/FedVaDE_source/dataset
PY=/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$CAMPAIGN/logs"
cd "$CODE_ROOT"

status_is() {
  "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == sys.argv[2]' "$1/result.json" "$2"
}

run_one() {
  local out="$1" expected="$2" label="$3"
  shift 3
  if [[ -e "$out" ]]; then
    status_is "$out" "$expected" || { echo "Existing output is incomplete: $out" >&2; return 1; }
    echo "SKIP $label"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  echo "START $label $(date -Iseconds)"
  "$PY" scripts/run_experiment.py "$@" --data-dir "$DATA" --out "$out" >"$CAMPAIGN/logs/$label.log" 2>&1 || return 1
  status_is "$out" "$expected" || return 1
  echo "DONE $label $(date -Iseconds)"
}

evaluate_one() {
  local out="$1" population="$2" label="$3" target="$1/evaluation_$2.json"
  if [[ -e "$target" ]]; then
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["population"]==sys.argv[2] and "300" in d["rounds"]' "$target" "$population"
    echo "SKIP eval $label $population"
    return 0
  fi
  "$PY" vendor/audited_scripts/common/evaluate.py --run "$out" --data-dir "$DATA" --rounds 300 --population "$population" --inference-seeds 10 --output "$target" >"$CAMPAIGN/logs/${label}_eval_${population}.log" 2>&1 || return 1
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["population"]==sys.argv[2] and "300" in d["rounds"]' "$target" "$population" || return 1
  echo "DONE eval $label $population $(date -Iseconds)"
}

pipeline() {
  local ds="$1" seed="$2" alpha="$3" tag="${1}_s${2}"
  local prepared="$CAMPAIGN/prepared/$ds/s$seed"
  local a="$CAMPAIGN/runs/$ds/A/s$seed"
  local b="$CAMPAIGN/runs/$ds/B/s$seed"
  local c="$CAMPAIGN/runs/$ds/C/s$seed"
  local common=(--dataset "$ds" --seed "$seed" --alpha "$alpha")
  run_one "$prepared" prepared "${tag}_prepare" --group prepare "${common[@]}" || return 1
  run_one "$a" ok "${tag}_A" --group A "${common[@]}" --prepared "$prepared/pre_gmm.pt" || return 1
  run_one "$b" ok "${tag}_B" --group B "${common[@]}" --prepared "$prepared/pre_gmm.pt" --dp-config "configs/formal_${ds}.json" || return 1
  if [[ "${#groups[@]}" == 3 ]]; then
    run_one "$c" ok "${tag}_C" --group C "${common[@]}" --prepared "$prepared/pre_gmm.pt" --b-run "$b" || return 1
  fi
  for group in "${groups[@]}"; do
    local out="$CAMPAIGN/runs/$ds/$group/s$seed"
    evaluate_one "$out" unique_full "${tag}_${group}" || return 1
    evaluate_one "$out" legacy_clients "${tag}_${group}" || return 1
  done
}

pids=()
for ds in iris breast_cancer banknote_authentication; do
  alpha=0.05
  [[ "$ds" == iris ]] && alpha=0.01
  for seed in "${seeds[@]}"; do
    pipeline "$ds" "$seed" "$alpha" >"$CAMPAIGN/logs/${ds}_s${seed}_pipeline.log" 2>&1 &
    pids+=("$!")
  done
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo failed >"$CAMPAIGN/campaign.exit"
  exit 1
fi
echo complete >"$CAMPAIGN/campaign.exit"
