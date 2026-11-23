#!/bin/zsh
# Multi-CA 实时监控：默认每 5 秒刷新一次，直到 Ctrl-C。
# 使用 --once 只查询一次。
set -u
KEY="$HOME/.ssh/id_ed25519_autodl_fuxian_33990"
HOST="root@connect.cqa1.seetacloud.com"
PORT="39225"
REMOTE="bash /root/autodl-tmp/fedvade_project/experiments/multica_factorial_v1/monitoring/remote_status.sh"
INTERVAL="${MULTICA_MONITOR_INTERVAL:-5}"
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

if [ ! -f "$KEY" ]; then
  echo "SSH key not found: $KEY"
  echo "Press Enter to close."
  read -r _
  exit 1
fi

trap 'echo; echo "监控已停止。"; exit 0' INT TERM
echo "Multi-CA 实时监控已启动：每 ${INTERVAL} 秒刷新；Ctrl-C 停止。"
sleep 1

while true; do
  printf '\033[2J\033[H'
  echo "Multi-CA 实时监控 | $(date '+%Y-%m-%d %H:%M:%S')"
  echo "刷新间隔：${INTERVAL}s | Ctrl-C 停止"
  echo
  if ssh -n -i "$KEY" -p "$PORT" -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$REMOTE"; then
    ssh_rc=0
  else
    ssh_rc=$?
    echo
    echo "SSH 查询失败（exit ${ssh_rc}）。检查网络、Host、端口和 key。"
  fi
  if [ "$ONCE" -eq 1 ]; then
    exit "$ssh_rc"
  fi
  echo
  echo "下一次刷新：${INTERVAL}s 后。Ctrl-C 停止。"
  sleep "$INTERVAL"
done
