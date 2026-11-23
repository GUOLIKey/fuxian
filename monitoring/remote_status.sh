#!/usr/bin/env bash
# Read-only live monitor for the 12-run 2x2 warm-start experiment.
set -u
ROOT=/root/autodl-tmp/fedvade_project
EXP="$ROOT/experiments/multica_pretrained_2x2_v1"
PY="$ROOT/.venv-cpu/bin/python"
INTERVAL=5
WATCH=0
[ "${1:-}" = "--watch" ] && WATCH=1
[ "${2:-}" != "" ] && INTERVAL="$2"

render() {
  printf '\033[2J\033[H'
  echo "Multi-CA 2x2 Warm-Start | 12 runs | NEW server"
  echo "Server: root@connect.cqa1.seetacloud.com:10959 | $(date -u +%FT%TZ)"
  echo
  "$PY" - "$EXP" <<'PY'
import json, sys
from pathlib import Path
exp = Path(sys.argv[1])
runs = []
for model in ('AUTHOR', 'F01'):
    for seed in (0, 1, 2):
        for agg in ('off', 'on'):
            d = exp / 'runs' / f'{model}_seed{seed}_{agg}'
            s = d / 'status.json'
            obj = {}
            if s.exists():
                try: obj = json.loads(s.read_text())
                except Exception: obj = {'status': 'UNREADABLE'}
            runs.append((model, seed, agg, obj))
counts = {'RUNNING': 0, 'DONE': 0, 'ERROR': 0, 'PENDING': 0}
for _, _, _, o in runs:
    st = o.get('status', 'PENDING')
    counts[st] = counts.get(st, 0) + 1
print(f"Runs: {len(runs)} | RUNNING {counts.get('RUNNING',0)} | DONE {counts.get('DONE',0)} | ERROR {counts.get('ERROR',0)} | PENDING {counts.get('PENDING',0)}")
print()
print("Model  Seed Agg  Round   ARI     NLL     K    GenMMD   GenSWD   Status")
print("-----  ---- ---  -----  ------  ------  ---  -------  -------  ------")
for model, seed, agg, o in runs:
    st = o.get('status', 'PENDING')
    latest = o.get('latest') or {}
    if st == 'DONE' and isinstance(latest.get('client_record_metrics_10seed'), dict):
        ari = latest['client_record_metrics_10seed']['ari']['mean']
        nll = latest.get('nll_unique')
        k = latest.get('k_micro')
        mmd = latest.get('gen_mmd_mean')
        swd = latest.get('gen_swd_mean')
        rnd = o.get('round', 300)
    else:
        ari = latest.get('ari')
        nll = latest.get('server_nll')
        k = latest.get('server_K')
        mmd = None
        swd = None
        rnd = o.get('round', 0)
    def f(x, w=6, p=4):
        return '—'.rjust(w) if x is None else f'{float(x):.{p}f}'.rjust(w)
    print(f"{model:5}  {seed:4} {agg:3}  {str(rnd):>5}  {f(ari)}  {f(nll)}  {str(k if k is not None else '—'):>3}  {f(mmd,7)}  {f(swd,7)}  {st}")
print()
errors = []
for model, seed, agg, o in runs:
    if o.get('last_error'):
        errors.append(f"{model}/seed{seed}/{agg}: {o['last_error']}")
if errors:
    print("Last errors:")
    for e in errors[:8]: print(' ', e)
else:
    print("Last errors: none")
PY
  echo
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
      --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
  else
    echo "GPU: nvidia-smi unavailable"
  fi
  echo "CPU/RAM: $(nproc) vCPU | $(free -h | awk '/^Mem:/ {print $3 " used / " $2}')"
  echo "Disk: $(df -h /root/autodl-tmp | awk 'NR==2 {print $4 " free / " $2 " (" $5 " used)"}')"
  echo
  echo "Processes: $(ps -eo cmd | grep -c '[t]rain_runner.py') train_runner | $(ps -eo cmd | grep -c '[r]un_all.sh') launcher"
  echo
  echo "Recent errors in run logs:"
  found=0
  for f in "$EXP"/logs/*.log; do
    [ -f "$f" ] || continue
    if grep -qE "Traceback|RuntimeError|CUDA error|nan|NaN|ERROR" "$f" 2>/dev/null; then
      found=1
      echo "--- $(basename "$f")"
      grep -E "Traceback|RuntimeError|CUDA error|nan|NaN|ERROR" "$f" | tail -n 3
    fi
  done
  [ "$found" -eq 1 ] || echo "none"
  echo
  echo "Latest run log:"
  latest_log=$(ls -t "$EXP"/logs/*.log 2>/dev/null | head -1 || true)
  if [ -n "$latest_log" ]; then
    echo "--- $(basename "$latest_log")"
    tail -c 1600 "$latest_log" | tr '\r' '\n' | tail -n 6
  else
    echo "no logs"
  fi
  echo
  echo "Next refresh: ${INTERVAL}s | Ctrl-C to stop"
}

if [ "$WATCH" -eq 1 ]; then
  while true; do render; sleep "$INTERVAL"; done
else
  render
fi
