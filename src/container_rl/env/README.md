# Container-RL 🚢📦

JAX-based Gymnasium reinforcement learning environment for the board game **Container**.

## Overview

Container is a board game about shipping logistics and market speculation. Players own factories producing coloured containers, which they can sell to other players' harbours, ship to auction island, and deliver to their island store for scoring. The game ends when the second container colour runs out.

This project implements the game rules as a **Gymnasium functional JAX environment**, enabling GPU-accelerated RL training.

## Files

| File | Description |
|------|-------------|
| `container_rules.md` | Full board game rules summary (from BGG) |
| `container.py` | Complete refactor with ActionEncoder (11 action types, ~2500 discrete actions), multi-player state, auction mechanics, and encoding/decoding tests |

## Architecture

- **`ContainerFunctional`** — Functional JAX env implementing `FuncEnv` interface (pure transition/observation/reward functions)
- **`ContainerJaxEnv`** — Gymnasium wrapper for standard `env.step()` / `env.reset()` API
- **`ActionEncoder`** — Encodes/decodes between discrete action indices and semantic action parameters (opponent, colour, price slot, etc.)
- **`EnvState`** — NamedTuple of JAX arrays for the full game state (per-player cash, loans, factories, warehouses, stores, ships, auction state)

## Action Space (v2)

~2500 discrete actions covering:

| Actions | Count |
|---------|-------|
| Buy factory (choose colour) | 5 |
| Buy warehouse | 1 |
| Produce containers | 1 |
| Buy from opponent's factory store | (n_players-1) × colours × price slots |
| Move ship to harbour + load | (n_players-1) × colours × price slots |
| Move ship to open sea | 1 |
| Move ship to auction island | 1 |
| Pass | 1 |
| Take loan | 1 |
| Repay loan | 1 |
| Domestic sale | 2 × colours × price slots |

## Status

- ✅ State definition, action encoding/decoding
- ✅ Initial state setup (factories, cash, secret value cards)
- ⏳ Game logic (transition function with all rules)
- ⏳ Observation vector (flattened state)
- ⏳ Reward function (net worth change)
- ⏳ Opponent AI (random policy)
- ⏳ Training pipeline (RL agent)

## Requirements

- Python 3.10+
- `jax`, `jaxlib`
- `gymnasium`
- `flax`
- `pygame` (for rendering)

## Running

```python
from container_rl.env.container import ContainerJaxEnv

env = ContainerJaxEnv(num_players=2, num_colors=5)
obs, info = env.reset()
```
