#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/fedvade_project/reproduction_runs/round5_broadcast
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python
SOURCE=/root/autodl-tmp/fedvade_project/fuxian-round2/FedVaDE_source
trap 'printf "FAILED %s\n" "$(date -Is)" > state.txt' ERR
printf 'RUNNING author %s\n' "$(date -Is)" > state.txt
"$PY" runner.py fedvade --source "$SOURCE" --out formal_author_seed0 --seed 0 --device cpu --schedule author > formal_author_seed0.log 2>&1
printf 'RUNNING post_server %s\n' "$(date -Is)" > state.txt
"$PY" runner.py fedvade --source "$SOURCE" --out formal_post_server_seed0 --seed 0 --device cpu --schedule post_server > formal_post_server_seed0.log 2>&1
printf 'DONE %s\n' "$(date -Is)" > state.txt
