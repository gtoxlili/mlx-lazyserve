#!/usr/bin/env bash
# Deploy mlx-lazyserve from this SOURCE repo into a production RUNTIME entity that lives
# OUTSIDE ~/Downloads (a macOS TCC-protected folder). The launchd agent then runs the
# runtime venv's python directly — no `uv run` wrapper, no TCC "Operation not permitted",
# no authorization popup. Source stays here for dev; the running service is the runtime.
#
#   cp deploy/service.env.example deploy/service.env   # then fill in API key etc.
#   bash deploy/install.sh                             # build + (re)install the service
#
# Idempotent — re-run after code changes to redeploy. No sudo required.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${MLX_LAZYSERVE_HOME:-$HOME/.mlx-lazyserve}"
LABEL="dev.influo.mlx-lazyserve"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UV="$(command -v uv || true)"
[ -n "$UV" ] || { echo "error: uv not found in PATH"; exit 1; }

ENV_FILE="$SRC/deploy/service.env"
[ -f "$ENV_FILE" ] || { echo "error: missing $ENV_FILE — copy deploy/service.env.example and fill it in"; exit 1; }

case "$RUNTIME" in
  "$HOME"/Downloads/*|"$HOME"/Documents/*|"$HOME"/Desktop/*)
    echo "error: runtime $RUNTIME is in a TCC-protected folder — pick a different MLX_LAZYSERVE_HOME"; exit 1;;
esac

echo "==> sync source -> runtime: $RUNTIME"
mkdir -p "$RUNTIME/logs"
rsync -a --delete \
  --exclude '.git/' --exclude '.venv/' --exclude 'logs/' \
  --exclude '*.local.plist' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.maintenance' --exclude 'deploy/service.env' \
  "$SRC"/ "$RUNTIME"/

echo "==> build runtime venv (uv sync --extra vision)"
"$UV" sync --project "$RUNTIME" --extra vision

echo "==> render LaunchAgent from deploy/service.env"
set -a; . "$ENV_FILE"; set +a
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>${LABEL}</string>
    <!-- Runtime venv entry, run directly (non-protected dir -> no TCC, no uv, no popup). -->
    <key>ProgramArguments</key>
    <array>
        <string>${RUNTIME}/.venv/bin/mlx-lazyserve</string>
    </array>
    <key>WorkingDirectory</key><string>${RUNTIME}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MLX_LAZYSERVE_HOST</key><string>${MLX_LAZYSERVE_HOST:-0.0.0.0}</string>
        <key>MLX_LAZYSERVE_PORT</key><string>${MLX_LAZYSERVE_PORT:-41434}</string>
        <key>MLX_LAZYSERVE_IDLE_TIMEOUT</key><string>${MLX_LAZYSERVE_IDLE_TIMEOUT:-600}</string>
        <key>MLX_LAZYSERVE_MAX_TOKENS</key><string>${MLX_LAZYSERVE_MAX_TOKENS:-8192}</string>
        <key>MLX_LAZYSERVE_WIRED_LIMIT_MB</key><string>${MLX_LAZYSERVE_WIRED_LIMIT_MB:-0}</string>
        <key>MLX_LAZYSERVE_API_KEYS</key><string>${MLX_LAZYSERVE_API_KEYS:-}</string>
        <key>HF_HUB_DISABLE_XET</key><string>${HF_HUB_DISABLE_XET:-1}</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ProcessType</key><string>Interactive</string>
    <key>ThrottleInterval</key><integer>10</integer>
    <key>StandardOutPath</key><string>${RUNTIME}/logs/stdout.log</string>
    <key>StandardErrorPath</key><string>${RUNTIME}/logs/stderr.log</string>
</dict>
</plist>
PLIST

echo "==> (re)bootstrap LaunchAgent"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "==> done. runtime=$RUNTIME"
echo "    logs:   tail -f $RUNTIME/logs/stderr.log"
echo "    health: curl http://127.0.0.1:${MLX_LAZYSERVE_PORT:-41434}/health"
