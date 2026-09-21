#!/usr/bin/env bash
#
# deploy.sh — build the frontend and push backend + frontend to the server.
#
# Runs from a workstation (node lives here, not necessarily on the server).
#
#   ./deploy.sh                  # build, push everything, restart, health check
#   ./deploy.sh --backend        # backend .py only
#   ./deploy.sh --frontend       # build + push frontend only
#   ./deploy.sh --no-build       # push an existing frontend/dist
#   ./deploy.sh --restart        # restart the service and health check only
#   ./deploy.sh --dry-run        # print what would happen
#
# Override any of these from the environment:
#   HOST=aibox REMOTE=/home/ai/scripts/rag SERVICE=doc-rag-api PORT=8088
#
set -euo pipefail

HOST="${HOST:-aibox}"
REMOTE="${REMOTE:-/home/ai/scripts/rag}"
SERVICE="${SERVICE:-doc-rag-api}"
PORT="${PORT:-8088}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DO_BACKEND=1 DO_FRONTEND=1 DO_BUILD=1 DO_RESTART=1 DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)   DO_FRONTEND=0 DO_BUILD=0 ;;
    --frontend)  DO_BACKEND=0 ;;
    --no-build)  DO_BUILD=0 ;;
    --restart)   DO_BACKEND=0 DO_FRONTEND=0 DO_BUILD=0 ;;
    --dry-run)   DRY=1 ;;
    -h|--help)   sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *)           echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if (( DRY )); then printf '   [dry] %s\n' "$*"; else "$@"; fi; }

# ------------------------------------------------------------------ preflight

command -v rsync >/dev/null || die "rsync not found"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$HOST" true 2>/dev/null \
  || die "cannot ssh to '$HOST' without a password prompt — check ~/.ssh/config"

say "target $HOST:$REMOTE (service $SERVICE, port $PORT)"

# -------------------------------------------------------------------- backend

if (( DO_BACKEND )); then
  say "pushing backend"
  # Explicit file list, never --delete: the server's backend/ also holds
  # outputs/, chunks.jsonl and log/ from ingest runs, and doc-rag.env is the
  # LIVE config — copying the repo template over it would clobber the server's
  # settings.
  run rsync -az --info=NAME \
    "$REPO"/backend/api.py \
    "$REPO"/backend/ingest.py \
    "$REPO"/backend/chunker.py \
    "$REPO"/backend/clean_det.py \
    "$REPO"/backend/infer.py \
    "$HOST:$REMOTE/backend/"
fi

# ------------------------------------------------------------------- frontend

if (( DO_FRONTEND )); then
  if (( DO_BUILD )); then
    command -v npm >/dev/null || die "npm not found; use --no-build"
    say "building frontend"
    if [[ -f "$REPO/frontend/package-lock.json" ]]; then
      run npm --prefix "$REPO/frontend" ci
    else
      run npm --prefix "$REPO/frontend" install
    fi
    run npm --prefix "$REPO/frontend" run build
  fi

  [[ -f "$REPO/frontend/dist/index.html" ]] || (( DRY )) \
    || die "frontend/dist/index.html missing — run without --no-build"

  say "pushing frontend"
  # Stage then swap: a partially-synced dist/ would serve an index.html that
  # references assets which are not there yet. The staging dir also avoids the
  # classic dist/dist nesting from 'scp -r dist remote:.../frontend/'.
  run rsync -az --delete --info=stats1 \
    "$REPO/frontend/dist/" "$HOST:$REMOTE/frontend/.dist-staging/"
  run ssh "$HOST" "set -e
    cd '$REMOTE/frontend'
    rm -rf dist.old
    [ -d dist ] && mv dist dist.old
    mv .dist-staging dist
    rm -rf dist.old"
fi

# -------------------------------------------------------------------- restart

if (( DO_RESTART )); then
  say "restarting $SERVICE"
  # -t for the sudo password prompt; harmless if sudo is passwordless.
  run ssh -t "$HOST" "sudo systemctl restart '$SERVICE'"

  if (( ! DRY )); then
    say "waiting for health"
    for i in $(seq 1 20); do
      if ssh "$HOST" "curl -sf http://127.0.0.1:$PORT/api/health" >/tmp/dr-health 2>/dev/null; then
        chunks=$(python3 -c 'import json,sys;print(json.load(open("/tmp/dr-health"))["collection"]["points"])' 2>/dev/null || echo '?')
        say "healthy — $chunks chunks indexed"

        # The frontend mount is decided at import; surface which branch it took.
        ssh "$HOST" "journalctl -u '$SERVICE' -n 40 --no-pager \
          | grep -Ei 'serving frontend|no frontend' | tail -1" || true

        rm -f /tmp/dr-health
        exit 0
      fi
      sleep 1
    done
    rm -f /tmp/dr-health
    warn "health check did not pass in 20s — inspect with:"
    warn "  ssh $HOST 'journalctl -u $SERVICE -n 40 --no-pager'"
    exit 1
  fi
fi

say "done"
