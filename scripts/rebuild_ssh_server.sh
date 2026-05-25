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
PROD_COPY="/guy/code/container-prod"

echo "=== Container RL — Deploy with AI ==="
echo "  Branch dir: $PROJECT_DIR"
echo "  Prod copy:  $PROD_COPY"
echo "  Model:      $MODEL_PATH"
echo "  Port:       22 (standard SSH)"

# ── 1. Stop running services ──────────────────────────────────────────────
echo ""
echo "[1/5] Stopping services..."
if [[ -n "$SSH_SERVICE" ]]; then
    sudo systemctl stop "$SSH_SERVICE" 2>/dev/null || true
fi
if [[ -n "$PYTHON_SERVICE" ]]; then
    sudo systemctl stop "$PYTHON_SERVICE" 2>/dev/null || true
fi
sudo pkill -f "container_rl.server" 2>/dev/null || true
sudo pkill -f "container-rl-ssh" 2>/dev/null || true
sleep 1

# ── 2. Rebuild Go SSH server ──────────────────────────────────────────────
echo ""
echo "[2/5] Building Go SSH server..."
cd "$GO_SSH_DIR"
sudo go build -o "$GO_BINARY" ./cmd/container-rl-ssh/
echo "  Built: $GO_BINARY"

# ── 3. Copy source files to prod directory ────────────────────────────────
echo ""
echo "[3/5] Copying Python source to prod..."
sudo rsync -a --delete \
    "$PROJECT_DIR/container_rl/" \
    "$PROD_COPY/container_rl/" \
    --exclude '__pycache__' \
    --exclude '*.pyc'
echo "  Copied $PROJECT_DIR/container_rl/ → $PROD_COPY/container_rl/"

# ── 4. Update systemd service to use this model ───────────────────────────
echo ""
echo "[4/5] Updating systemd services..."
OVERRIDE_DIR="/etc/systemd/system/$PYTHON_SERVICE.service.d"
sudo mkdir -p "$OVERRIDE_DIR"
sudo tee "$OVERRIDE_DIR/ai-model.conf" > /dev/null <<EOF
[Service]
ExecStart=
ExecStart=/opt/container-rl/.venv/bin/python3 -m container_rl.server \\
    --host 127.0.0.1 \\
    --port 9876 \\
    --db /opt/container-rl/container_server.db \\
    --maintainer-token cb3f4f5b5e9a0cb3 \\
    --ai-model $MODEL_PATH
EOF
echo "  Wrote $OVERRIDE_DIR/ai-model.conf"

# Also fix the SSH service to use port 22
SSH_OVERRIDE_DIR="/etc/systemd/system/$SSH_SERVICE.service.d"
sudo mkdir -p "$SSH_OVERRIDE_DIR"
sudo tee "$SSH_OVERRIDE_DIR/port-22.conf" > /dev/null <<EOF
[Service]
ExecStart=
ExecStart=/opt/container-rl/container-rl-ssh --addr :22 --host-key-dir /opt/container-rl/ssh_host_keys --keys /opt/container-rl/ssh_keys.json --game-addr 127.0.0.1:9876 --maintainer-token cb3f4f5b5e9a0cb3
EOF
echo "  Wrote $SSH_OVERRIDE_DIR/port-22.conf"
sudo systemctl daemon-reload

# ── 5. Start services ──────────────────────────────────────────────────────
echo ""
echo "[5/5] Starting services..."
if [[ -n "$PYTHON_SERVICE" ]]; then
    sudo systemctl start "$PYTHON_SERVICE"
    echo "  Started: $PYTHON_SERVICE"
else
    nohup python -m container_rl.server \
        --host 0.0.0.0 \
        --port 9876 \
        --ai-model "$MODEL_PATH" \
        > /tmp/container-server.log 2>&1 &
    echo "  Started (PID $!): logs at /tmp/container-server.log"
fi
sleep 1

if [[ -n "$SSH_SERVICE" ]]; then
    sudo systemctl start "$SSH_SERVICE"
    echo "  Started: $SSH_SERVICE"
else
    nohup "$GO_BINARY" \
        --addr :22 \
        --host-key-dir "$PROJECT_DIR/ssh_host_keys" \
        --keys "$PROJECT_DIR/ssh_keys.json" \
        --game-addr 127.0.0.1:9876 \
        > /tmp/container-ssh.log 2>&1 &
    echo "  Started (PID $!): logs at /tmp/container-ssh.log"
fi

echo ""
echo "=== Done ==="
echo "  SSH server on :22 (standard port — just 'ssh play-container.tech')"
echo "  Game server on :9876"
echo "  AI model: $MODEL_PATH"
echo ""
echo "Sources are now synced from $PROJECT_DIR → $PROD_COPY"
echo "Run 'ssh play-container.tech' to connect."
