# Container RL — SSH Server Deployment

## Architecture

```
User ──SSH (port 22)──→ Go SSH Server (container-rl-ssh)
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
       Wish SSH            Bubble Tea          Go TCP Client
       (auth, keys)        (TUI)               (connects to
           │                  │                  localhost:9876)
           ▼                  ▼                  ▼
    ssh_keys.json            PTY           Python Game Server
    (key store)                              (container-server)
                                                  │
                                                  ▼
                                            SQLite + JAX
```

- **Go SSH server** — Handles SSH connections, authenticates users via public keys or password (anonymous), renders the Bubble Tea TUI, and connects to the Python game engine.
- **Python game server** — Runs the Container board game logic using JAX. Listens on `127.0.0.1:9876`. Go connects as a client over TCP using length-prefixed JSON.
- **Shared state** — SSH public keys stored in `ssh_keys.json`. Game state persisted in SQLite (`container_server.db`).

## Auth Flow

| Connection method | Session behavior |
|---|---|
| Registered SSH key | Auto-login as your username → Main menu |
| Unknown SSH key | Prompt: "New key! Choose a username:" → registered → Main menu |
| No key (password) | Disconnected with instructions: "Generate a key first: `ssh-keygen -t ed25519`" |

The SSH key is the sole identity mechanism. No passwords needed — if the user has a key in `~/.ssh/`, the SSH client sends it automatically. On the first connection it's captured and bound to a username. Every subsequent connection from that machine auto-logs in.

## Prerequisites

- **Server:** Linux VPS (any provider — DigitalOcean, Hetzner, etc.)
- **Go 1.22+** (to build the SSH server)
- **Python 3.12+** + JAX (for the game engine)
- **Systemd** (for production process management)
- **iptables** or **nftables** (for port 22 → 2222 redirect)

## Building from source

All commands assume you start in the repository root (`/guy/code/container-rl-website` in development, or wherever you cloned it on the VPS).

### Go SSH server

The Go project lives in the `container-rl-ssh/` subdirectory. The compiled binary is produced inside that directory.

```bash
# cd into the Go project
cd container-rl-ssh
go mod tidy
go build -o container-rl-ssh ./cmd/container-rl-ssh/
cd ..
# Binary is now: container-rl-ssh/container-rl-ssh
```

The resulting binary is static (~8 MB) and requires no runtime dependencies.

### Python game server

The Python project uses `pyproject.toml` at the repo root. Run this from the repo root:

```bash
# cd back to repo root (if you cd'd into container-rl-ssh above)
cd /opt/container-rl          # production, or
cd /path/to/repo              # development

python3 -m venv .venv
.venv/bin/pip install -e .    # installs container-rl with all deps
```

## Deployment

### Quick start (one command)

The setup script handles everything automatically — builds the Go binary, installs Python deps, copies files, and starts services:

```bash
# cd to repo root, then:
sudo bash deploy/setup.sh
```

### Manual steps

If you prefer to run each step individually, or need to customize the deployment:

#### 0. Build (prerequisite for the steps below)

```bash
# cd to repo root, then:
cd container-rl-ssh && go build -o container-rl-ssh ./cmd/container-rl-ssh/ && cd ..
python3 -m venv .venv && .venv/bin/pip install -e .
```

#### 1. Create the service user

```bash
useradd --system --user-group --create-home container-rl
```

### 2. Set up the application directory

```bash
mkdir -p /opt/container-rl/ssh_host_keys
chown -R container-rl:container-rl /opt/container-rl
chmod 700 /opt/container-rl/ssh_host_keys
```

### 3. Copy files

Run from the repo root:

```bash
# Binary (built above in container-rl-ssh/ directory)
cp container-rl-ssh/container-rl-ssh /opt/container-rl/

# Systemd units (in deploy/ directory)
cp deploy/container-rl-server.service /etc/systemd/system/
cp deploy/container-rl-ssh.service    /etc/systemd/system/
```

If you built the Python virtualenv elsewhere (not in `/opt/container-rl`), symlink it:

```bash
ln -s /path/to/.venv /opt/container-rl/.venv
```

### 4. Install systemd units

```bash
systemctl daemon-reload
systemctl enable --now container-rl-server.service   # Python game engine
systemctl enable --now container-rl-ssh.service       # Go SSH server
```

### 5. Configure port forwarding

The SSH server listens on port 2222 by default. To allow users to type `ssh <hostname>` without specifying a port, redirect port 22 to 2222:

```bash
# Stop any existing SSH server on port 22 first:
systemctl stop sshd
systemctl disable sshd

# Redirect 22 → 2222
iptables -t nat -A PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222

# persist across reboots (Debian/Ubuntu)
apt install iptables-persistent
iptables-save > /etc/iptables/rules.v4
```

### 6. DNS

Add an A record pointing your domain to the server's public IP:

```
container-rl.example.com.  A  1.2.3.4
```

### 7. Verify

From any machine with an SSH key:

```bash
ssh container-rl.example.com
# → New key detected! Choose a username: ...
```

Or test locally while still on the server:

```bash
ssh -p 2222 localhost
```

## Configuration

### Go SSH server flags

| Flag | Default | Description |
|---|---|---|
| `--addr` | `:2222` | SSH listen address |
| `--host-key-dir` | `./ssh_host_keys` | Directory for SSH host keys |
| `--keys` | `ssh_keys.json` | JSON file for registered SSH public keys |
| `--game-addr` | `127.0.0.1:9876` | Python game server address |

### Python game server flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | TCP listen address (use `127.0.0.1` in production) |
| `--port` | `9876` | TCP listen port |
| `--db` | `container_server.db` | SQLite database path |

### Environment variables

None required. The internal Go ↔ Python auth uses localhost detection — connections from `127.0.0.1` are trusted (no password check needed).

## Managing the service

```bash
# View logs
journalctl -u container-rl-ssh -f
journalctl -u container-rl-server -f

# Restart
systemctl restart container-rl-ssh
systemctl restart container-rl-server

# Stop
systemctl stop container-rl-ssh
systemctl stop container-rl-server
```

## Key store format

`ssh_keys.json` is an array of key entries:

```json
[
  {
    "player_name": "alice",
    "player_id": 1,
    "public_key": "ssh-ed25519 AAAAC3NzaC1l...",
    "fingerprint": "SHA256:...",
    "created_at": "2026-05-11T12:00:00Z"
  }
]
```

Keys are added automatically during signup. The file is safe to back up.

## Troubleshooting

| Problem | Fix |
|---|---|
| `go build` takes forever | The Charm libraries compile quickly. If using `modernc.org/sqlite` (not in this project), it compiles SQLite from scratch. |
| SSH connection refused | Check `systemctl status container-rl-ssh` — ensure the service is running. Try `ss -tlnp \| grep 2222`. |
| "Cannot connect to game server" in TUI | Ensure Python server is running: `systemctl status container-rl-server`. Verify `--game-addr` points to the correct host/port. |
| Blank screen after login | Bubble Tea needs a proper TERM. Ensure your SSH client sends `TERM=xterm-256color`. |
| Port 22 already in use | Stop OpenSSH: `systemctl stop sshd`. Or use a different port for the Go server. |
| Host key warnings on reconnect | The host key directory changes each deploy. Add to `~/.ssh/config`: `StrictHostKeyChecking accept-new`. |
