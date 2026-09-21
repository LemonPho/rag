#!/usr/bin/env bash
#
# deploy.sh — build the frontend and restart the services.
#
#   ./deploy.sh                # build frontend, restart api, health check
#   ./deploy.sh --pull         # git pull first
#   ./deploy.sh --deps         # also sync python deps into the venv
#   ./deploy.sh --no-build     # skip the frontend build
#   ./deploy.sh --restart      # restart + health check only
#   ./deploy.sh --all          # restart qdrant too (rarely needed)
#   ./deploy.sh --dry-run      # print what would happen
#
# Override from the environment:
#   SERVICE=doc-rag-api PORT=8088 VENV=/home/ai/venvs/rag
#
set -euo pipefail

SERVICE="${SERVICE:-doc-rag-api}"
PORT="${PORT:-8088}"
VENV="${VENV:-/home/ai/venvs/rag}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$REPO/frontend"

DO_PULL=0 DO_BUILD=1 DO_DEPS=0 DO_RESTART=1 RESTART_ALL=0 DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull)     DO_PULL=1 ;;
    --deps)     DO_DEPS=1 ;;
    --no-build) DO_BUILD=0 ;;
    --restart)  DO_BUILD=0 ;;
    --all)      RESTART_ALL=1 ;;
    --dry-run)  DRY=1 ;;
    -h|--help)  sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *)          echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n'  "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n'  "$*" >&2; exit 1; }
run()  { if (( DRY )); then printf '   [dry] %s\n' "$*"; else "$@"; fi; }

say "repo $REPO — service $SERVICE on port $PORT"

# ----------------------------------------------------------------------- pull

if (( DO_PULL )); then
  say "git pull"
  # Local edits to backend/infer.py (NO_REPEAT_NGRAM_SIZE=0 and the removed
  # images_config field) must be committed, or a pull will refuse or clobber.
  if ! (( DRY )) && [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
    git -C "$REPO" status --short
    die "working tree is dirty — commit or stash before pulling"
  fi
  run git -C "$REPO" pull --ff-only
fi

# ----------------------------------------------------------------------- deps

if (( DO_DEPS )); then
  [[ -x "$VENV/bin/pip" ]] || die "no venv at $VENV"
  say "syncing python deps"
  run "$VENV/bin/pip" install --quiet --upgrade fastapi uvicorn qdrant-client
fi

# -------------------------------------------------------------------- frontend

if (( DO_BUILD )); then
  command -v npm >/dev/null || die "npm not found on this host — install node, \
or build on a workstation and use --no-build"

  say "building frontend"
  if [[ -f "$FRONTEND/package-lock.json" ]]; then
    run npm --prefix "$FRONTEND" ci
  else
    run npm --prefix "$FRONTEND" install
  fi

  # Build into a staging dir, then swap. Building straight into dist/ empties it
  # first, so a failed build would leave the site serving nothing.
  run npm --prefix "$FRONTEND" run build -- --outDir dist.new --emptyOutDir

  if ! (( DRY )); then
    [[ -f "$FRONTEND/dist.new/index.html" ]] \
      || die "build produced no index.html — see the output above"
    rm -rf "$FRONTEND/dist.old"
    [[ -d "$FRONTEND/dist" ]] && mv "$FRONTEND/dist" "$FRONTEND/dist.old"
    mv "$FRONTEND/dist.new" "$FRONTEND/dist"
    rm -rf "$FRONTEND/dist.old"
    say "frontend built ($(du -sh "$FRONTEND/dist" | cut -f1))"
  fi
fi

# --------------------------------------------------------------------- restart

if (( DO_RESTART )); then
  units=("$SERVICE")
  (( RESTART_ALL )) && units=(qdrant.service "$SERVICE")

  say "restarting ${units[*]}"
  run sudo systemctl restart "${units[@]}"

  if (( DRY )); then say "done (dry run)"; exit 0; fi

  say "waiting for health"
  for _ in $(seq 1 20); do
    if body=$(curl -sf "http://127.0.0.1:$PORT/api/health" 2>/dev/null); then
      chunks=$(printf '%s' "$body" \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["collection"]["points"])' \
        2>/dev/null || echo '?')
      say "healthy — $chunks chunks indexed"

      # The static mount is decided once, at import. Surface which branch it
      # took so a missing frontend is not a silent 404.
      journalctl -u "$SERVICE" -n 40 --no-pager 2>/dev/null \
        | grep -Ei 'serving frontend|no frontend' | tail -1 || true
      exit 0
    fi
    sleep 1
  done

  warn "health check did not pass in 20s:"
  warn "  journalctl -u $SERVICE -n 40 --no-pager"
  exit 1
fi

say "done"
