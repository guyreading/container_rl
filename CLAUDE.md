# CLAUDE.md

Guidance for working in this repo. Read this before diving in — it points at the
detailed docs that already exist rather than repeating them.

## What this is

Container RL Env simulates the board game *Container* (shipping logistics / market
speculation) as a JAX/Gymnasium RL environment, with a terminal UI for humans to
play locally or over the network (via a Python TCP server and, for public internet
play, a Go SSH gateway that gives each player a shell-free `ssh play-container.tech`
experience).

Two languages, one repo:
- `container_rl/` — the actual project: env, RL training, TUI, multiplayer server. Published to PyPI as `container-rl`.
- `container-rl-ssh/` — a Go SSH gateway (Wish + Bubble Tea) that authenticates players by SSH key and proxies them into the Python TCP server. Only relevant for the public multiplayer deployment; not needed for local dev.

## Where things live

| Concern | File(s) |
|---|---|
| Game rules, JAX env, action space, observation space | `container_rl/env/container.py` — **read `container_rl/env/README.md` first**, it's a thorough architecture doc (multi-head action space, recurrent shopping, masks, obs layout) |
| Human-readable board game rules | `container_rl/env/container_rules.md` |
| Action space encoding reference (per-action head/mask tables) | `container_rl/env/container_action_space.md` |
| Single-player local TUI (`container_rl play`) | `container_rl/cli.py` |
| Multiplayer TUI client (`container-client`) | `container_rl/client/tui.py` |
| Low-level TCP client used by the TUI | `container_rl/client/client.py` |
| Admin/ops TUI for listing & managing live games | `container_rl/client/maintainer.py` |
| Multiplayer TCP server (`container-server`) | `container_rl/server/server.py` (socket/select loop) → `container_rl/server/game_manager.py` (game lifecycle, turn orchestration) → `container_rl/server/database.py` (SQLite persistence) |
| Wire protocol (length-prefixed JSON + pickled `EnvState`) | `container_rl/server/protocol.py` |
| RL training entry point (PPO via sb3-contrib `MaskablePPO`) | `container_rl/train.py` |
| Self-play opponent pool, ELO tracking | `container_rl/self_play.py` |
| Go SSH gateway (deployment, auth flow, systemd units) | `container-rl-ssh/` — **read `container_rl/server/README.md`**, despite the path it documents the *Go* deployment layer, not the Python server internals |
| Go SSH gateway source | `container-rl-ssh/internal/{server,tui,auth,client,db,protocol}` |

There is currently **no single doc describing how `server.py` / `game_manager.py` /
`database.py` / `protocol.py` compose**, nor one covering `cli.py` / `client/tui.py` /
`self_play.py` / `train.py`. If you spend real time understanding one of those,
consider leaving a short module-level docstring or README behind for next time.

## Dev workflow

No `justfile` (removed deliberately in commit `6659a2b`) — use `uv` directly. Note
that some sandboxes (including this one at times) don't have `uv` on PATH; if so,
these need to run wherever `uv` is actually available.

```bash
uv sync                          # install deps (dev group by default)
uv run container_rl play -p 3 -h 0,1,2   # local single-machine TUI
uv run container-server          # multiplayer TCP server
uv run container-client          # multiplayer TUI client
uv run ruff format --check .     # formatting (CI-enforced)
uv run ruff check .              # lint (CI-enforced)
uv run ty check .                # type check (CI-enforced)
uv run pytest                    # tests
uv run --group docs zensical serve   # live-reload docs preview
```

CI (`.github/workflows/ci.yml`) runs lint, `ty` type-check, pytest across Python
3.12–3.14, and a coverage gate (`fail_under = 50` in `pyproject.toml`).

Go side: `cd container-rl-ssh && go build -o container-rl-ssh ./cmd/container-rl-ssh/`.

## Known gaps / debt (as of 2026-07-19)

- **`cli.py` vs `client/tui.py` duplication**: the single-player TUI (`cli.py`) and
  multiplayer TUI client (`client/tui.py`) independently reimplement nearly
  identical Rich rendering and raw-terminal input handling — colour helpers, store/
  island/ship renderers, the player-card renderer, and the produce/buy-factory/
  buy-from-factory/move-load submenus. E.g. `_player_card` in `cli.py:188` and
  `client/tui.py:195` are essentially the same function. This is the biggest real
  refactor opportunity (~400-500 duplicated lines, risk of the two UIs drifting),
  but it touches interactive terminal code that can't be verified without a live
  TTY session with multiple players, so it wasn't attempted blind — do it with a
  human driving the TUI to confirm behavior, not headless.
- **`tests/container_rl/test_cli.py` is empty** — `cli.py` (~1200 lines, the
  single-player TUI) has no test coverage at all.
- **`docs/usage.md` and `docs/index.md`** are still the mkdocs-template boilerplate
  (`docs/usage.md` just says `import container_rl`) — inaccurate, since the actual
  usage is the `container_rl play` / `container-server` / `container-client` CLI
  commands, not library import. Worth rewriting from the README's Quick Start.
- **Release flow is partially broken**: `CONTRIBUTING.md` documents bumping the
  version and writing `CHANGELOG/<version>.md`, then running `scripts/release.py`
  (a PEP 723 script — run via `uv run scripts/release.py`) which reads that file.
  The `CHANGELOG/` directory doesn't currently exist in the repo (deleted in
  `6659a2b`), so a release will fail at `notes_path.read_text()` until it's
  recreated with at least one entry.
- Zero TODO/FIXME comments and no commented-out code blocks anywhere in
  `container_rl/` — the codebase is otherwise clean; don't expect to find debt
  markers, look for structural issues instead.

## Conventions

- `container_rl/env/container.py` is written as a **pure functional JAX env**
  (`ContainerFunctional`) wrapped by a thin Gymnasium-compatible class
  (`ContainerJaxEnv`). Game logic changes belong in the functional core; keep it
  `jax.jit`-compatible (no Python-side branching on traced values, no I/O).
- The action space is fixed at **5 heads** regardless of player/colour count
  (`num_heads()` always returns 5); only per-head *sizes* vary — see
  `container_rl/env/README.md` before changing anything action-space-related.
- Server and client speak **length-prefixed JSON** (`server/protocol.py`), with
  `EnvState` itself sent as a separately pickled blob inside the JSON payload.
  `deserialize_state()` fills missing fields with zeros for backward compat with
  older saves — if you add a field to `EnvState`, add it to `_STATE_KEYS` too.
