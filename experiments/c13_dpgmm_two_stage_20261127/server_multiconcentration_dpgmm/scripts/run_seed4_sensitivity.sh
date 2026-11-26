#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=/root/autodl-tmp/fedvade_project/reproduction_runs/c13_server_dpgmm_multigranularity_v1
CAMPAIGN="$BASE/sensitivity_iris_banknote_s4_20261127"
DATA=/root/autodl-tmp/fedvade_project/fuxian-round2/FedVaDE_source/dataset
PY=/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python
CONFIG="$CODE_ROOT/configs/server_dpgmm.json"
MANIFEST="$CODE_ROOT/SOURCE_MANIFEST.json"
SEED=4

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$CAMPAIGN/logs"
cd "$CODE_ROOT"

for dataset in iris banknote_authentication; do
  "$PY" scripts/precheck.py verify \
    --dataset "$dataset" \
    --receipt "$CODE_ROOT/prechecks/$dataset.json" \
    --config "$CONFIG" --manifest "$MANIFEST"
done

run_one() {
  local dataset="$1" alpha="$2" tag="${1}_s${SEED}"
  local out="$CAMPAIGN/runs/$dataset/s$SEED"
  "$PY" scripts/run_experiment.py \
    --dataset "$dataset" --seed "$SEED" --alpha "$alpha" \
    --data-dir "$DATA" --server-config "$CONFIG" --out "$out" \
    >"$CAMPAIGN/logs/${tag}.log" 2>&1
  "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "ok"' "$out/result.json"
  for population in unique_full legacy_clients; do
    local target="$out/evaluation_${population}.json"
    "$PY" vendor/audited_scripts/common/evaluate.py \
      --run "$out" --data-dir "$DATA" --rounds 300 \
      --population "$population" --inference-seeds 10 --output "$target" \
      >"$CAMPAIGN/logs/${tag}_eval_${population}.log" 2>&1
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["population"]==sys.argv[2] and "300" in d["rounds"] and len(d["rounds"]["300"]["draws"])==10' "$target" "$population"
  done
}

run_one iris 0.01 >"$CAMPAIGN/logs/iris_s4_pipeline.log" 2>&1 &
iris_pid=$!
run_one banknote_authentication 0.05 >"$CAMPAIGN/logs/banknote_authentication_s4_pipeline.log" 2>&1 &
banknote_pid=$!

failed=0
wait "$iris_pid" || failed=1
wait "$banknote_pid" || failed=1
if (( failed )); then
  echo failed >"$CAMPAIGN/campaign.exit"
  exit 1
fi
echo complete >"$CAMPAIGN/campaign.exit"
