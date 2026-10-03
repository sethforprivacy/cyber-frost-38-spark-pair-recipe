#!/usr/bin/env bash
# Bring the CyberFrost TP2 pair up from rank 0's host: the minimal form of the boot owner we run under systemd
# (no sudo needed for the launch itself), e.g. from rank 0's crontab:
#
#   @reboot sleep 60 && bash /path/to/recipe/launcher/pair-up.sh up >> $HOME/cyberfrost-run/owner.log 2>&1
#
#   pair-up.sh up       1. exits at once if $RUN_DIR/DISABLED exists
#                       2. waits for rank 1 (ssh), one ping per fabric rail (LINK_CHECKS) and > 100 GiB MemAvailable
#                          on both ranks; with DUAL_RAIL=1 a dead second rail falls back to rail 1 only (RoCEnante off)
#                       3. vllm-rank.sh up on rank 1, then rank 0, with the profile (each with its host memory guard)
#                       4. waits up to 30 min for :$PORT/health while both rank containers keep running, then writes
#                          $RUN_DIR/ready
#   pair-up.sh check    prints both ranks' docker commands (vllm-rank.sh check); starts nothing
#   pair-up.sh down     saves each rank's container log to $RUN_DIR/logs/, then removes the rank containers
#   pair-up.sh status   one line per rank: uptime s, MemAvailable GiB, free 32 MiB blocks, running containers
#
# Env: PROFILE (default launcher/profile.env), HOSTS_ENV (default launcher/hosts.env), RUN_DIR ($HOME/cyberfrost-run),
# REMOTE_DIR (this recipe's path on rank 1; default: the same path as here).
# Profile lines are KEY=VALUE, one per line, comments on their own lines (no inline comments).
set -uo pipefail
MODE=${1:-up}
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=load-hosts.sh
. "$HERE/load-hosts.sh"
PROFILE=${PROFILE:-$HERE/profile.env}
RUN_DIR=${RUN_DIR:-$HOME/cyberfrost-run}
REMOTE_DIR=${REMOTE_DIR:-$(cd "$HERE/.." && pwd)}
NAME=${CONTAINER:-vllm_cyberfrost}
mkdir -p "$RUN_DIR/logs"
log() { echo "$(date -u -Iseconds) $*"; }
r() { if [ "$1" = 0 ]; then bash -c "$2"; else ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(rank_ssh "$1")" "$2"; fi; }
[ -s "$PROFILE" ] || { log "FATAL: profile $PROFILE missing"; exit 1; }
KN=""; while IFS= read -r kv; do case "$kv" in ''|'#'*) ;; *) KN+=" $(printf %q "$kv")";; esac; done < "$PROFILE"
PORT=$(sed -n 's/^PORT=//p' "$PROFILE"); PORT=${PORT:-8901}

case "$MODE" in
 check)
  for i in 1 0; do echo "# rank $i"; r "$i" "env$KN bash $REMOTE_DIR/launcher/vllm-rank.sh $i check"; done
  exit;;
 down)
  ts=$(date -u +%Y%m%d-%H%M%S)
  for i in 0 1; do
    r "$i" "docker logs $NAME" > "$RUN_DIR/logs/$NAME-rank$i-$ts.log" 2>&1 || true
    r "$i" "docker rm -f $NAME >/dev/null 2>&1; true"
  done
  rm -f "$RUN_DIR/ready"; log "pair down; logs in $RUN_DIR/logs/*-$ts.log"; exit;;
 status)
  st='echo "$(awk "{print int(\$1)}" /proc/uptime) $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) $(awk "/Normal/{print \$NF}" /proc/buddyinfo) $(docker ps -q --filter name='"$NAME"' | wc -l)"'
  for i in 0 1; do echo "$i $(r "$i" "$st" 2>/dev/null || echo down)"; done
  exit;;
 up) ;;
 *) echo "usage: pair-up.sh {up|check|down|status}" >&2; exit 2;;
esac

[ -f "$RUN_DIR/DISABLED" ] && { log "owner disabled ($RUN_DIR/DISABLED); not starting"; exit 0; }
log "=== owner start ==="
rm -f "$RUN_DIR/ready"
deadline=$(( $(date +%s) + 900 ))
until r 1 true 2>/dev/null; do [ "$(date +%s)" -lt "$deadline" ] || { log "FATAL: rank 1 unreachable"; exit 1; }; sleep 10; done
RAIL_ENV=""
for hop in ${LINK_CHECKS:?LINK_CHECKS not set (hosts.env)}; do
  rail=${hop%%:*}; ip=${hop#*:}; n=0
  until r 0 "ping -c1 -W2 $ip >/dev/null"; do
    n=$((n + 1))
    if [ "$n" -ge 36 ]; then
      [ "$rail" = 1 ] && { log "FATAL: fabric rail 1 ($ip) down"; exit 1; }
      log "WARN: fabric rail $rail ($ip) down: starting on rail 1 only, RoCEnante off"; RAIL_ENV=" DUAL_RAIL=0"; break
    fi
    sleep 5
  done
done
for i in 0 1; do
  n=0; until r "$i" "awk '/MemAvailable/{exit !(\$2 > 100*1048576)}' /proc/meminfo"; do
    n=$((n + 1)); [ "$n" -lt 60 ] || { log "FATAL: rank $i MemAvailable stays < 100 GiB"; exit 1; }; sleep 5; done
done
log "ranks, fabric and memory ready; profile:$KN$RAIL_ENV"
for i in 1 0; do
  r "$i" "env$KN$RAIL_ENV bash $REMOTE_DIR/launcher/vllm-rank.sh $i up >/dev/null" || { log "FATAL: vllm rank $i up"; exit 1; }
done
log "ranks started, memory guards armed; waiting for :$PORT/health"
deadline=$(( $(date +%s) + 1800 ))
until curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null; do
  [ "$(date +%s)" -lt "$deadline" ] || { log "FATAL: :$PORT not healthy after 30 min"; exit 1; }
  for i in 0 1; do
    [ "$(r "$i" "docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null")" = true ] || { log "FATAL: rank $i container not running"; exit 1; }
  done
  sleep 15
done
date -u -Iseconds > "$RUN_DIR/ready"
log "=== serving on :$PORT ==="
