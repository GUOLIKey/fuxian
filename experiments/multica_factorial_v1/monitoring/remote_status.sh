#!/usr/bin/env bash
# Read-only remote monitor for Multi-CA Stage A. Never modifies training.
set -u
ROOT=/root/autodl-tmp/fedvade_project
BASE="$ROOT/experiments/multica_factorial_v1"
RUNS="$ROOT/reproduction_runs/multica_factorial_v1"
PY="$ROOT/.venv/bin/python"

echo "FedVaDE + Multi-CA | Remote Monitor"
echo "Server: $(hostname) | $(date -u +%FT%TZ)"
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

"$PY" - "$BASE/experiment_manifest.json" "$RUNS" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
runs_root = Path(sys.argv[2])
try:
    manifest = json.loads(manifest_path.read_text())
except Exception:
    manifest = {}

print("GROUP   SEED   STATUS       ROUND     ARI        NMI        NLL")
completed = running = failed = 0
for group in "ABCD":
    for seed in (0, 1, 2):
        out = runs_root / "stage_a" / f"{group}_seed{seed}"
        def load(name):
            try:
                return json.loads((out / name).read_text())
            except Exception:
                return {}
        status = load("status.json")
        result = load("result.json")
        progress = load("progress.json")
        if not out.exists():
            state = "PENDING"
        elif result.get("status") == "ok":
            state = "COMPLETED"
            completed += 1
        elif result.get("status") == "failed":
            state = "FAILED"
            failed += 1
        elif status.get("status") in ("RUNNING", "INTERRUPTED"):
            state = status["status"]
        else:
            state = "INTERRUPTED"
        if state == "RUNNING":
            running += 1
        last = result.get("last", {}) if state == "COMPLETED" else {}
        ari = last.get("unique_ari", progress.get("unique_ari"))
        nmi = last.get("unique_nmi", progress.get("unique_nmi"))
        nll = last.get("server_gmm_nll", progress.get("server_gmm_nll"))
        round_no = progress.get("round", status.get("round", 0)) or 0
        def fmt(value):
            return "NA" if value is None else f"{float(value):.4f}"
        print(f"{group:<7} {seed:<6} {state:<12} {round_no:>4}/300  "
              f"{fmt(ari):<10} {fmt(nmi):<10} {fmt(nll):<10}")
print(f"\nCompleted: {completed} / 12")
print(f"Running:   {running}")
print(f"Failed:    {failed}")
github = manifest.get("github", {})
print(f"GitHub:    {github.get('status','NOT_PUSHED')} "
      f"{github.get('branch','')} {github.get('commit','')}")
PY

echo
echo "Running processes:"
ps -eo pid,etime,cmd | grep -E "run_factorial.py|run_stage_a.sh|pretrain_shared.py" | grep -v grep || true
echo
echo "tmux sessions:"
tmux ls 2>/dev/null || echo "none"
echo
echo "Recent non-empty error logs:"
found=0
for f in "$RUNS"/logs/*.err.log; do
  [ -s "$f" ] || continue
  found=1
  echo "--- $(basename "$f")"
  tail -c 1500 "$f" | tr "\r" "\n" | tail -n 5
done
[ "$found" -eq 1 ] || echo "none"
echo
echo "Latest log tail:"
latest=$(ls -t "$RUNS"/logs/[A-D]_seed*.log 2>/dev/null | head -1 || true)
[ -n "$latest" ] || latest=$(ls -t "$RUNS"/logs/run_stage_a.log 2>/dev/null | head -1 || true)
if [ -n "$latest" ]; then
  echo "--- $(basename "$latest")"
  tail -c 1800 "$latest" | tr "\r" "\n" | tail -n 8
else
  echo "no logs"
fi
