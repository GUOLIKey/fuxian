#!/bin/zsh
# 2x2 warm-start live monitor (NEW server).
# Double-click once: one SSH connection, remote panel refreshes every 5 seconds.
# Ctrl-C stops. `--once` performs one query. Read-only: never starts/stops jobs.

set -u
KEY="$HOME/.ssh/id_ed25519_autodl_fuxian_33990"
HOST="root@connect.cqa1.seetacloud.com"
PORT="10959"
REMOTE_BASE="bash /root/autodl-tmp/fedvade_project/experiments/multica_pretrained_2x2_v1/monitoring/remote_status.sh"
INTERVAL="${MULTICA_MONITOR_INTERVAL:-5}"
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

if [ ! -f "$KEY" ]; then
  echo "SSH key not found: $KEY" >&2
  echo "Press Enter to close."; read -r _
  exit 1
fi

trap 'echo; echo "监控已停止。"; exit 0' INT TERM
echo "Connecting to ${HOST}:${PORT} ..."
echo "Panel refreshes every ${INTERVAL}s; Ctrl-C stops."

if [ "$ONCE" -eq 1 ]; then
  ssh -i "$KEY" -p "$PORT" -o BatchMode=yes -o ConnectTimeout=15 \
    -o ServerAliveInterval=15 "$HOST" "$REMOTE_BASE"
  exit $?
fi

while true; do
  ssh -tt -i "$KEY" -p "$PORT" -o BatchMode=yes -o ConnectTimeout=15 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    "$HOST" "$REMOTE_BASE --watch $INTERVAL"
  ssh_rc=$?
  echo
  echo "SSH disconnected (exit ${ssh_rc}); reconnecting in 5s. Ctrl-C stops."
  sleep 5
done
