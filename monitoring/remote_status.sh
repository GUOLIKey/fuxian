#!/usr/bin/env bash
# Read-only live panel for the corrected baselines (A/C inference, B/D blocked).
set -u
ROOT=/root/autodl-tmp/fedvade_project
EXP="$ROOT/experiments/multica_pretrained_corrected_v1"
PY="$ROOT/.venv-cpu/bin/python"
INTERVAL=5
WATCH=0
[ "${1:-}" = "--watch" ] && WATCH=1
[ "${2:-}" != "" ] && INTERVAL="$2"

render() {
  printf '\033[2J\033[H'
  echo "FedVaDE corrected 2x2 | A/C direct inference | B/D blocked"
  echo "Server: root@connect.cqa1.seetacloud.com:10959 | $(date -u +%FT%TZ)"
  echo
  "$PY" - "$EXP" <<'PY'
import json
import sys
from pathlib import Path

exp = Path(sys.argv[1])
path = exp / 'results' / 'baselines.json'
data = None
if path.exists():
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        data = {'error': repr(exc)}

def f(x, w=8, p=4):
    return '—'.rjust(w) if x is None else f'{float(x):.{p}f}'.rjust(w)

print("Grp Model         Mode                Status    ARI_s0    ARI_s1    ARI_s2"
      "   ARI_0-2   ARI_0-9      NLL    GenMMD    GenSWD")
print("--- ------------  ------------------  --------  --------  --------  --------"
      "  --------  --------  -------  --------  --------")
if data and 'models' in data:
    by = {r['group']: r for r in data['models']}
    for grp, model in (('A', 'Author .pk'), ('C', 'F01 ep100')):
        r = by.get(grp)
        if not r:
            print(f"{grp:3} {model:12}  direct inference    PENDING"
                  + "".rjust(78))
            continue
        s = r['actual_ari_seeds_0_1_2']
        print(f"{grp:3} {model:12}  direct inference    DONE    "
              f"{f(s[0])}  {f(s[1])}  {f(s[2])}  "
              f"{f(r['actual_ari_mean_0_2'])}  {f(r['actual_ari_mean_0_9'])}  "
              f"{f(r['nll_unique'],7,3)}  {f(r['gen_mmd_mean'],8,5)}  "
              f"{f(r['gen_swd_mean'],8,5)}")
else:
    print("Evaluation results not available yet: "
          + (data.get('error') if data else 'baselines.json missing'))
for grp, model in (('B', 'Author'), ('D', 'F01')):
    print(f"{grp:3} {model:12}  Multi-CA            BLOCKED"
          + ''.rjust(80))
print()
if data and 'models' in data:
    by = {r['group']: r for r in data['models']}
    for grp in ('A', 'C'):
        r = by.get(grp)
        if r:
            print(f"Group {grp}: historical {r['historical_ari_reference']:.4f}"
                  f" | actual(0-9) {r['actual_ari_mean_0_9']:.4f}"
                  f" | delta "
                  f"{r['actual_ari_mean_0_9'] - r['historical_ari_reference']:+.4f}"
                  f" | Purity {r['client_purity_mean']:.4f}"
                  f" | NMI {r['client_nmi_mean']:.4f}"
                  f" | K={r['server_K']}")
    pc = data.get('protocol_check', {})
    print(f"F01 protocol check: exact={pc.get('exact')} "
          f"(reproduced {pc.get('F01_reproduced_mean')})")
print("B/D: both payloads are server-only; no client states exist.")
print("     No reconstruction, no continuation, no training. See audit/BLOCKER_B_D.md.")
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
  if [ -f "$EXP/logs/eval.log" ]; then
    echo "Latest eval log:"
    tail -n 4 "$EXP/logs/eval.log"
  else
    echo "Latest eval log: none"
  fi
  echo
  echo "Recent errors:"
  found=0
  for log in "$EXP"/logs/*.log; do
    [ -f "$log" ] || continue
    if grep -qE "Traceback|RuntimeError|FAILED|Error" "$log" 2>/dev/null; then
      found=1
      echo "--- $(basename "$log")"
      grep -E "Traceback|RuntimeError|FAILED|Error" "$log" | tail -n 3
    fi
  done
  [ "$found" -eq 1 ] || echo "none"
  echo
  echo "Next refresh: ${INTERVAL}s | Ctrl-C to stop"
}

if [ "$WATCH" -eq 1 ]; then
  while true; do render; sleep "$INTERVAL"; done
else
  render
fi
