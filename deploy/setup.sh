#!/bin/bash
# deploy/setup.sh — system-level deployment for Container RL SSH server.
# Run as root (sudo). Assumes Go binary and Python venv are already built.
# See container_rl/server/README.md for the full build + deploy guide.
set -euo pipefail

APP_DIR="/opt/container-rl"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Container RL SSH Server Setup ==="
echo "  App dir:   $APP_DIR"
echo "  Repo root: $REPO_ROOT"
echo ""

# ── 1. Verify prerequisites ────────────────────────────────────────────
BIN="$REPO_ROOT/container-rl-ssh/container-rl-ssh"
if [ ! -f "$BIN" ]; then
    echo "ERROR: Go binary not found at $BIN"
    echo "  Build it first: cd container-rl-ssh && go build -o container-rl-ssh ./cmd/container-rl-ssh/"
    exit 1
fi

if [ ! -d "$APP_DIR/.venv" ] && [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "ERROR: Python virtualenv not found."
    echo "  Create it first: python3 -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

# ── 2. Create system user ──────────────────────────────────────────────
echo "[1/4] Creating service user ..."
if ! id -u container-rl &>/dev/null; then
    useradd --system --user-group --create-home container-rl
    echo "  Created container-rl user."
else
    echo "  User container-rl already exists."
fi

# ── 3. Create directories and copy files ────────────────────────────────
echo "[2/4] Copying files ..."
mkdir -p "$APP_DIR/ssh_host_keys"
chmod 700 "$APP_DIR/ssh_host_keys"

cp "$BIN" "$APP_DIR/"
chown -R container-rl:container-rl "$APP_DIR"

cp "$REPO_ROOT/deploy/container-rl-server.service" /etc/systemd/system/
cp "$REPO_ROOT/deploy/container-rl-ssh.service"    /etc/systemd/system/
chmod 644 /etc/systemd/system/container-rl-server.service
chmod 644 /etc/systemd/system/container-rl-ssh.service
echo "  Done."

# ── 4. Symlink Python venv if it's in the repo (not in /opt) ────────────
if [ ! -d "$APP_DIR/.venv" ] && [ -d "$REPO_ROOT/.venv" ]; then
    echo "[3/4] Symlinking Python venv ..."
    ln -sf "$REPO_ROOT/.venv" "$APP_DIR/.venv"
    echo "  Linked $REPO_ROOT/.venv → $APP_DIR/.venv"
fi

# ── 5. Install and start systemd units ──────────────────────────────────
echo "[4/4] Enabling services ..."
systemctl daemon-reload
systemctl enable --now container-rl-server.service
systemctl enable --now container-rl-ssh.service
echo "  Done."

echo ""
echo "=== Setup complete ==="
echo "  SSH:  ssh -p 2222 localhost"
echo ""
echo "For port 22 (no -p flag):"
echo "  iptables -t nat -A PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222"
echo ""
echo "Users connect: ssh <your-server-ip>"
