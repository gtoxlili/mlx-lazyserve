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
        <key>MLX_LAZYSERVE_ENABLE_THINKING</key><string>${MLX_LAZYSERVE_ENABLE_THINKING:-false}</string>
        <key>MLX_LAZYSERVE_KV_BITS</key><string>${MLX_LAZYSERVE_KV_BITS:-0}</string>
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
DOMAIN="gui/$(id -u)"
PORT="${MLX_LAZYSERVE_PORT:-41434}"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
# bootout is ASYNCHRONOUS: bootstrapping before launchd finishes DEREGISTERING the old job
# fails with "Bootstrap failed: 5: Input/output error" and leaves the service UNLOADED (the
# port freeing first is not enough — launchd's domain state lags). So wait until the job is
# GONE from launchd (`launchctl print` fails) AND the port is free, force-killing a straggler
# that overstays ~15s (slow uvicorn drain / big-model unload).
for i in $(seq 1 80); do
    deregistered=1; launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 && deregistered=0
    pid="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1)"
    [ "$deregistered" = 1 ] && [ -z "$pid" ] && break
    [ -n "$pid" ] && [ "$i" -ge 30 ] && kill -9 "$pid" 2>/dev/null || true
    sleep 0.5
done
bootstrapped=0
for _ in $(seq 1 20); do
    if launchctl bootstrap "$DOMAIN" "$PLIST" 2>/dev/null; then bootstrapped=1; break; fi
    sleep 0.5
done
if [ "$bootstrapped" -ne 1 ]; then
    echo "error: launchctl bootstrap kept failing; last error below:"
    launchctl bootstrap "$DOMAIN" "$PLIST" || true
    exit 1
fi
launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || { echo "error: job not running after bootstrap"; exit 1; }

echo "==> done. runtime=$RUNTIME"
echo "    logs:   tail -f $RUNTIME/logs/stderr.log"
echo "    health: curl http://127.0.0.1:${MLX_LAZYSERVE_PORT:-41434}/health"
