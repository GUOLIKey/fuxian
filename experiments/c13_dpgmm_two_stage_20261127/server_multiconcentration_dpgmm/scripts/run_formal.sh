#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=/root/autodl-tmp/fedvade_project/reproduction_runs/c13_server_dpgmm_multigranularity_v1
TRAINING_SEED="${TRAINING_SEED:-}"
if [[ ! "$TRAINING_SEED" =~ ^[0-4]$ ]]; then
  echo "Set exactly one TRAINING_SEED in 0..4; inference randomness is handled by --inference-seeds 10." >&2
  exit 2
fi
CAMPAIGN="$BASE/formal_server_dpgmm_s${TRAINING_SEED}_20261127"
DATA=/root/autodl-tmp/fedvade_project/fuxian-round2/FedVaDE_source/dataset
PY=/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python
CONFIG="$CODE_ROOT/configs/server_dpgmm.json"
MANIFEST="$CODE_ROOT/SOURCE_MANIFEST.json"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$CAMPAIGN/logs"
cd "$CODE_ROOT"

for dataset in iris breast_cancer banknote_authentication; do
  "$PY" scripts/precheck.py verify \
    --dataset "$dataset" \
    --receipt "$CODE_ROOT/prechecks/$dataset.json" \
    --config "$CONFIG" \
    --manifest "$MANIFEST"
done

run_one() {
  local dataset="$1" seed="$2" alpha="$3"
  local tag="${dataset}_s${seed}"
  local out="$CAMPAIGN/runs/$dataset/s$seed"
  if [[ -e "$out" ]]; then
    "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "ok"' "$out/result.json"
  else
    "$PY" scripts/run_experiment.py \
      --dataset "$dataset" --seed "$seed" --alpha "$alpha" \
      --data-dir "$DATA" --server-config "$CONFIG" --out "$out" \
      >"$CAMPAIGN/logs/${tag}.log" 2>&1
  fi
  for population in unique_full legacy_clients; do
    local target="$out/evaluation_${population}.json"
    if [[ ! -e "$target" ]]; then
      "$PY" vendor/audited_scripts/common/evaluate.py \
        --run "$out" --data-dir "$DATA" --rounds 300 \
        --population "$population" --inference-seeds 10 --output "$target" \
        >"$CAMPAIGN/logs/${tag}_eval_${population}.log" 2>&1
    fi
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["population"]==sys.argv[2] and "300" in d["rounds"] and len(d["rounds"]["300"]["draws"])==10' "$target" "$population"
  done
}

pids=()
for dataset in iris breast_cancer banknote_authentication; do
  alpha=0.05
  [[ "$dataset" == iris ]] && alpha=0.01
  for seed in "$TRAINING_SEED"; do
    run_one "$dataset" "$seed" "$alpha" >"$CAMPAIGN/logs/${dataset}_s${seed}_pipeline.log" 2>&1 &
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
