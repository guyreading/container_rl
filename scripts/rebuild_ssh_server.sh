#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/runs/container-ppo/final_model}"
GO_SSH_DIR="$PROJECT_DIR/container-rl-ssh"
GO_BINARY="/opt/container-rl/container-rl-ssh"
SSH_SERVICE="container-rl-ssh"
PYTHON_SERVICE="container-rl-server"

echo "=== Container RL — Deploy with AI ==="
echo "  Project:  $PROJECT_DIR"
echo "  Model:    $MODEL_PATH"
echo "  SSH bin:  $GO_BINARY"

# ── 1. Stop running services ──────────────────────────────────────────────
echo ""
echo "[1/4] Stopping services..."
if [[ -n "$SSH_SERVICE" ]]; then
    systemctl stop "$SSH_SERVICE" 2>/dev/null || true
fi
if [[ -n "$PYTHON_SERVICE" ]]; then
    systemctl stop "$PYTHON_SERVICE" 2>/dev/null || true
fi
pkill -f "container_rl.server" 2>/dev/null || true
pkill -f "container-rl-ssh" 2>/dev/null || true
sleep 1

# ── 2. Rebuild Go SSH server ──────────────────────────────────────────────
echo ""
echo "[2/4] Building Go SSH server..."
cd "$GO_SSH_DIR"
go build -o "$GO_BINARY" ./cmd/container-rl-ssh/
echo "  Built: $GO_BINARY"

# ── 3. Start Python game server ───────────────────────────────────────────
echo ""
echo "[3/4] Starting Python game server..."
cd "$PROJECT_DIR"

if [[ -n "$PYTHON_SERVICE" ]]; then
    systemctl start "$PYTHON_SERVICE"
    echo "  Started via systemctl: $PYTHON_SERVICE"
else
    nohup python -m container_rl.server \
        --host 0.0.0.0 \
        --port 9876 \
        --ai-model "$MODEL_PATH" \
        > /tmp/container-server.log 2>&1 &
    echo "  Started (PID $!): logs at /tmp/container-server.log"
fi
sleep 1

# ── 4. Start Go SSH server ────────────────────────────────────────────────
echo ""
echo "[4/4] Starting Go SSH server..."
if [[ -n "$SSH_SERVICE" ]]; then
    systemctl start "$SSH_SERVICE"
    echo "  Started via systemctl: $SSH_SERVICE"
else
    nohup "$GO_BINARY" \
        --addr :2222 \
        --host-key-dir "$PROJECT_DIR/ssh_host_keys" \
        --keys "$PROJECT_DIR/ssh_keys.json" \
        --game-addr 127.0.0.1:9876 \
        > /tmp/container-ssh.log 2>&1 &
    echo "  Started (PID $!): logs at /tmp/container-ssh.log"
fi

echo ""
echo "=== Done ==="
echo "  SSH server on :2222"
echo "  Game server on :9876"
echo "  AI model: $MODEL_PATH"
