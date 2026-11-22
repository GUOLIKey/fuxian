#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/fedvade_project/reproduction_runs/round8_checkpoint_finetune || exit 2
PY=/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PY" -u task_round8.py > task.log 2>&1
code=$?
if [ "$code" -eq 0 ] && [ -s ROUND8_REPORT.md ] && [ -s finetune_metrics.csv ] && [ -s inference_repeats.csv ] && [ -s best_finetuned_checkpoint/server.pt ]; then
  printf '{"exit_code":0,"finished_at":"%s"}\n' "$(date -Is)" > task_exit_status.tmp
  mv task_exit_status.tmp task_exit_status.json
  touch completed.marker
  "$PY" - <<'PY'
import json, time
from pathlib import Path
p=Path('progress.json');d=json.loads(p.read_text());d.update(stage='FINISHED',status='FINISHED',updated_at=time.time());tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
PY
  exit 0
fi
if [ "$code" -eq 0 ]; then code=4; fi
printf '{"exit_code":%s,"finished_at":"%s"}\n' "$code" "$(date -Is)" > task_exit_status.tmp
mv task_exit_status.tmp task_exit_status.json
touch failed.marker
"$PY" - <<'PY'
import json, time
from pathlib import Path
p=Path('progress.json');d=json.loads(p.read_text()) if p.exists() else {};d.update(stage='FAILED',status='FAILED',updated_at=time.time());tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
PY
exit "$code"
