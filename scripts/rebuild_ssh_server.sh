#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROD_COPY="/guy/code/container-prod"
PROD_MODEL_DIR="$PROD_COPY/model"
PROD_MODEL="$PROD_MODEL_DIR/best_model"
GO_SSH_DIR="$PROJECT_DIR/container-rl-ssh"
GO_BINARY="/opt/container-rl/container-rl-ssh"
SSH_SERVICE="container-rl-ssh"
PYTHON_SERVICE="container-rl-server"

# Maintainer token — never hardcode this; it gates maintainer_list /
# maintainer_delete on the game server.  Supply it via the environment:
#   MAINTAINER_TOKEN=... sudo -E ./scripts/rebuild_ssh_server.sh
# or drop it in a root-only file at /etc/container-rl/maintainer-token.
TOKEN_FILE="${TOKEN_FILE:-/etc/container-rl/maintainer-token}"
if [[ -z "${MAINTAINER_TOKEN:-}" ]] && [[ -r "$TOKEN_FILE" ]]; then
    MAINTAINER_TOKEN="$(< "$TOKEN_FILE")"
fi
if [[ -z "${MAINTAINER_TOKEN:-}" ]]; then
    echo "ERROR: MAINTAINER_TOKEN is not set and $TOKEN_FILE is unreadable." >&2
    echo "  Set it explicitly:  MAINTAINER_TOKEN=... sudo -E $0" >&2
    exit 1
fi

# Auto-discover best model from training runs (most recently modified wins)
AUTO_MODEL=""
for candidate in "$PROJECT_DIR"/runs/*/best_model/best_model.zip; do
    if [[ -f "$candidate" ]] && [[ -z "$AUTO_MODEL" || "$candidate" -nt "$AUTO_MODEL" ]]; then
        AUTO_MODEL="$candidate"
    fi
done
if [[ -z "$AUTO_MODEL" ]] && [[ -f "$PROJECT_DIR/runs/long-run/final_model.zip" ]]; then
    AUTO_MODEL="$PROJECT_DIR/runs/long-run/final_model.zip"
fi
if [[ -z "$AUTO_MODEL" ]] && [[ -f "$PROJECT_DIR/runs/smoke-test/final_model.zip" ]]; then
    AUTO_MODEL="$PROJECT_DIR/runs/smoke-test/final_model.zip"
fi

MODEL_PATH="${MODEL_PATH:-$AUTO_MODEL}"

echo "=== Container RL — Deploy with AI ==="
echo "  Branch:   $PROJECT_DIR"
echo "  Prod:     $PROD_COPY"
echo "  Model:    $MODEL_PATH"
echo "  Port:     22 (standard SSH)"

# ── 1. Stop services ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Stopping services..."
sudo systemctl stop "$SSH_SERVICE" 2>/dev/null || true
sudo systemctl stop "$PYTHON_SERVICE" 2>/dev/null || true
sudo pkill -f "container-rl-ssh" 2>/dev/null || true
sudo pkill -f "container_rl.server" 2>/dev/null || true
sleep 1

# ── 2. Rebuild Go SSH server ──────────────────────────────────────────────
echo ""
echo "[2/6] Building Go SSH server..."
cd "$GO_SSH_DIR"
# Build as the invoking user (the Go toolchain does not need root); only the
# install into /opt needs elevation.
BUILD_TMP="$(mktemp -d)"
trap 'rm -rf "$BUILD_TMP"' EXIT
go build -o "$BUILD_TMP/container-rl-ssh" ./cmd/container-rl-ssh/
sudo install -m 755 "$BUILD_TMP/container-rl-ssh" "$GO_BINARY"
echo "  Built: $GO_BINARY"

sudo setcap 'cap_net_bind_service=+ep' "$GO_BINARY"
echo "  Granted port 22 binding capability"

# ── 3. Clear nftables redirect ────────────────────────────────────────────
echo ""
echo "[3/6] Clearing nftables redirect..."
sudo nft flush table inet container-redirect 2>/dev/null || true
echo "  Done"

# ── 4. Sync Python source to prod ─────────────────────────────────────────
echo ""
echo "[4/6] Syncing Python source to prod..."
sudo rsync -a --delete \
    "$PROJECT_DIR/container_rl/" \
    "$PROD_COPY/container_rl/" \
    --exclude '__pycache__' \
    --exclude '*.pyc'
echo "  Synced $PROJECT_DIR/container_rl/ → $PROD_COPY/container_rl/"

# ── 5. Copy trained model ─────────────────────────────────────────────────
echo ""
echo "[5/6] Copying trained model..."
sudo mkdir -p "$PROD_MODEL_DIR"
if [[ -f "$MODEL_PATH" ]]; then
    sudo cp "$MODEL_PATH" "$PROD_MODEL.zip"
    echo "  Copied $MODEL_PATH → $PROD_MODEL.zip"
else
    echo "  WARNING: No model found at $MODEL_PATH — AI disabled"
    echo "  Searched: $PROJECT_DIR/runs/*/best_model/best_model.zip"
fi

# ── 6. Update systemd and start services ──────────────────────────────────
echo ""
echo "[6/6] Updating systemd and starting..."

# Token lives in a root-only EnvironmentFile so it never lands in a
# world-readable unit file (or in git).
ENV_FILE="/etc/container-rl/maintainer.env"
sudo mkdir -p "$(dirname "$ENV_FILE")"
printf 'MAINTAINER_TOKEN=%s\n' "$MAINTAINER_TOKEN" | sudo tee "$ENV_FILE" > /dev/null
sudo chmod 600 "$ENV_FILE"
sudo chown root:root "$ENV_FILE"
echo "  Token → $ENV_FILE (mode 600)"

# Python server: add --ai-model
OVERRIDE_DIR="/etc/systemd/system/$PYTHON_SERVICE.service.d"
sudo mkdir -p "$OVERRIDE_DIR"
sudo tee "$OVERRIDE_DIR/ai-model.conf" > /dev/null <<EOF
[Service]
EnvironmentFile=$ENV_FILE
ExecStart=
ExecStart=/opt/container-rl/.venv/bin/python3 -m container_rl.server \\
    --host 127.0.0.1 \\
    --port 9876 \\
    --db /opt/container-rl/container_server.db \\
    --maintainer-token \${MAINTAINER_TOKEN} \\
    --ai-model $PROD_MODEL
EOF
echo "  Python server → $PROD_MODEL"

# SSH server: bind to port 22
SSH_OVERRIDE_DIR="/etc/systemd/system/$SSH_SERVICE.service.d"
sudo mkdir -p "$SSH_OVERRIDE_DIR"
sudo tee "$SSH_OVERRIDE_DIR/port-22.conf" > /dev/null <<EOF
[Service]
EnvironmentFile=$ENV_FILE
ExecStart=
ExecStart=/opt/container-rl/container-rl-ssh --addr :22 --host-key-dir /opt/container-rl/ssh_host_keys --keys /opt/container-rl/ssh_keys.json --game-addr 127.0.0.1:9876 --maintainer-token \${MAINTAINER_TOKEN}
EOF
echo "  SSH server → port 22"

sudo systemctl daemon-reload
sudo systemctl start "$PYTHON_SERVICE"
sudo systemctl start "$SSH_SERVICE"

echo ""
echo "=== Done ==="
echo "  SSH:  ssh play-container.tech"
echo "  Model: $PROD_MODEL.zip"
echo "  (Override model: MODEL_PATH=/path/to/model.zip sudo -E $0)"
