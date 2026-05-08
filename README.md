# Container RL Env

![PyPI version](https://img.shields.io/pypi/v/container-rl.svg)

An RL environment to simulate the board game Container and train agents in

* GitHub: https://github.com/guyreading/container-rl/
* PyPI package: https://pypi.org/project/container-rl/
* Created by: **[Guy Reading](https://guyreading.github.io/)** | GitHub https://github.com/guyreading | PyPI https://pypi.org/user/guyreading/
* Free software: MIT License

## Features

* Full JAX-accelerated Gymnasium environment for the Container board game
* **Single-player TUI** — play locally against AI opponents
* **Multiplayer server** — host games over TCP, connect from separate terminals or machines
* Recurrent actions: multi-container Produce, BuyFromFactory, and LoadShip
* Action masking via MaskablePPO for RL training
* Persistent game state (SQLite), rejoin mid-game with name + password

## Quick Start

### Local TUI (single machine)

```bash
uv run container_rl play -p 3 -h 0,1,2
```

- `-p` / `--players` — number of players
- `-h` / `--humans` — comma-separated human indices (0-based)
- `-c` / `--colors` — number of container colors (default 5)

### Multiplayer (TCP server)

**Terminal 1 — start the server:**

```bash
uv run container-server [--host 0.0.0.0] [--port 9876]
```

**Terminal 2, 3, … — each player connects:**

```bash
uv run container-client [--host <server-ip>] [--port 9876]
```

Each player sees a main menu: **Create Game** or **Join Game**.

**Creating a game** — enter your name, optional password, choose player count and colors. You get a shareable game code (e.g. `"FALCON-42"`). Wait in the lobby.

**Joining a game** — enter the game code shared by the creator. Wait in the lobby.

The game starts automatically when all slots are filled. Each player takes their turn in sequence; the TUI renders the full board state. Actions are dispatched to the server, which broadcasts updates to all clients.

**Reconnecting** — rejoin with the same game code, name, and password. Your player slot is restored.

## Multiplayer Architecture

```
┌──────────────────────────────────────────┐
│           Game Server (daemon)            │
│  ┌────────────┐  ┌──────────────────┐    │
│  │Game Manager│  │ TCP Handler      │    │
│  └─────┬──────┘  └────────┬─────────┘    │
│        └─────────┬────────┘              │
│        ┌─────────┴────────┐              │
│        │  SQLite Database  │              │
│        │  (state + players)│              │
│        └───────────────────┘              │
└──────────────────────────────────────────┘
     ▲                          ▲
     │ TCP + JSON               │
┌────┴─────┐             ┌──────┴─────┐
│Client TUI│             │ Client TUI │
│ Player 1 │             │  Player 2  │
└──────────┘             └────────────┘
```

The server owns the game state and is authoritative — clients are thin terminals that render the board and send action indices. All game logic runs server-side via `env.step()`. Game state is persisted to SQLite after every action, supporting disconnection and rejoin.

## TUI Controls

| Key | Action |
|---|---|
| `1` | Buy Factory |
| `2` | Buy Warehouse |
| `3` | Produce (per-color pricing + leave idle) |
| `4` | Buy from Factory (multi-container, harbour price $2–$6) |
| `5` | Load Ship (multi-container from harbour) |
| `6` | Move to Open Sea |
| `7` | Auction (recurrent bidding + seller accept/reject) |
| `0` / Space | Pass |
| `8` | Take Loan |
| `9` | Repay Loan |
| `←→` | Browse history |
| `q` | Quit |

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://guyreading.github.io/container_rl/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/container-rl.git
cd container-rl

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `container_rl`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

Container RL Env was created in 2026 by Guy Reading.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
