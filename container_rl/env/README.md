# Container-RL 🚢📦

JAX-based Gymnasium reinforcement learning environment for the board game **Container**.

## Overview

Container is a board game about shipping logistics and market speculation. Players own factories producing coloured containers, which they can sell to other players' harbours, ship to auction island, and deliver to their island store for scoring. The game ends when the second container colour runs out.

This project implements the game rules as a **Gymnasium functional JAX environment**, enabling GPU-accelerated RL training.

## Files

| File | Description |
|------|-------------|
| `container_rules.md` | Full board game rules summary (from BGG) |
| `container.py` | Functional JAX environment with multi-head action space, recurrent shopping, action masks, and full game logic |

## Architecture

- **`ContainerFunctional`** — Functional JAX env implementing `FuncEnv` interface (pure `transition` / `observation` / `reward` functions, `jax.jit`-compatible)
- **`ContainerJaxEnv`** — Gymnasium wrapper for standard `env.step()` / `env.reset()` API
- **`ActionEncoder`** — Legacy encoder/decoder between flat discrete action indices and semantic action parameters; kept for backward compatibility
- **`EnvState`** — NamedTuple of JAX arrays for the full game state (per-player cash, loans, factories, warehouses, stores, ships, auction state)

## Action Space — Multi-Head Design

The action space uses a **multi-head** architecture inspired by [settlers-rl](https://settlers-rl.github.io). Instead of a single flat discrete index, the action is represented as a fixed-size integer array where each element (head) handles one sub-decision. The overall action log-probability for PPO training is the sum of the log-probs from each relevant head.

### Why Multi-Head?

- **Generalisation** — Feedback from one action choice (e.g. buying blue containers) can reinforce the *colour* head independently of the *price* or *opponent* heads, improving sample efficiency.
- **Action masking** — The environment provides per-head binary masks; invalid options are zeroed out before softmax so the agent never wastes time on illegal actions.
- **Recurrent shopping** — Actions 3 and 4 can buy **multiple containers in a single action** (as the real rules allow). A single recurrent purchase head is unrolled over 10 fixed steps, with a STOP sentinel.

### Head Layout

The action tensor is an `int32` array of shape `(num_heads,)` where:

```
num_heads = NUM_FIXED_HEADS + MAX_PURCHASE_STEPS = 4 + 10 = 14
```

| Head index | Name | Size | Used by action types | Description |
|---|---|---|---|---|
| 0 | **Action Type** | 11 | all | One of: BuyFactory, BuyWarehouse, Produce, BuyFromFactoryStore, MoveLoad, MoveSea, Auction, Pass, TakeLoan, RepayLoan, DomesticSale |
| 1 | **Opponent** | `n_players − 1` | BuyFromFactoryStore (3), MoveLoad (4) | Which opponent to target (clockwise from acting player) |
| 2 | **Color** | `n_colors` | BuyFactory (0), DomesticSale (10) | Which container colour |
| 3 | **Price Slot** | `PRICE_SLOTS` (10) | DomesticSale (10) | Which price slot ($1–$10) to sell from |
| 4–13 | **Purchase Steps** | `n_colors × PRICE_SLOTS + 1` (51) each | BuyFromFactoryStore (3), MoveLoad (4) | Recurrent shopping: each step encodes `color × PRICE_SLOTS + price_slot`, or STOP at index `n_colors × PRICE_SLOTS` |

### Which Heads Are Active Per Action

| Action | Action Type | Opponent | Color | Price Slot | Purchase Steps |
|---|---|---|---|---|---|
| Buy Factory (0) | ✓ | — | ✓ | — | — |
| Buy Warehouse (1) | ✓ | — | — | — | — |
| Produce (2) | ✓ | — | — | — | — |
| Buy from Factory Store (3) | ✓ | ✓ | — | — | ✓ (recurrent) |
| Move to Harbour + Load (4) | ✓ | ✓ | — | — | ✓ (recurrent) |
| Move to Open Sea (5) | ✓ | — | — | — | — |
| Auction (6) | ✓ | — | — | — | — |
| Pass (7) | ✓ | — | — | — | — |
| Take Loan (8) | ✓ | — | — | — | — |
| Repay Loan (9) | ✓ | — | — | — | — |
| Domestic Sale (10) | ✓ | — | ✓ | ✓ | — |

For inactive heads the output is ignored; their log-prob contribution is masked to zero during training.

### Recurrent Shopping Detail

For actions **3** (Buy from Factory Store) and **4** (Move to Harbour + Load), the purchase steps are processed by `jax.lax.scan`. The scan:

1. Reads purchase step `i` from `action[4 + i]`.
2. Decodes it into `(color, price_slot)` or STOP.
3. If STOP or any constraint fails (no cash, no space, no stock), the scan stops and ignores remaining steps.
4. Otherwise, the container is purchased and the next step is processed.

This allows buying up to `warehouse_count` or `SHIP_CAPACITY` containers in a single action. During PPO rollout the policy samples purchase steps autoregressively; during training all steps are processed in parallel and their log-probs summed.

### Action Masks

`ContainerFunctional._action_masks(state, params)` returns per-head binary vectors:

| Key | Shape | Meaning |
|---|---|---|
| `action_type` | (11,) | 1 = action type is currently legal |
| `opponent` | (n_players−1,) | 1 = opponent has purchasable stock |
| `color` | (n_colors,) | 1 = colour is a valid choice |
| `price_slot` | (PRICE_SLOTS,) | 1 = own store has containers at that price |
| `purchase` | (51,) | 1 = (color, slot) combo is affordable & in stock somewhere; STOP always valid |

These masks are appended to the observation vector so an RL policy can apply them: `masked_logits = logits + log(mask)`, then softmax → probabilities.

## Observation Space

The observation is a flat `float32` vector of size:

```
obs_size = np * 4                    # cash, loans, warehouse_count, ship_location per player
         + np * nc * 2               # factory_colors, island_store
         + np * nc * PRICE_SLOTS * 2 # factory_store, harbour_store
         + np * SHIP_CAPACITY        # ship_contents
         + nc                        # container_supply
         + 4                         # turn_phase, current_player, game_over, actions_taken
         + np                        # secret_value_color per player
         + 5                         # auction_active, auction_seller, auction_cargo_count
         + mask_size                 # action_type(11) + opponent(np-1) + color(nc) + price_slot(10) + purchase(51)
```

For 2 players, 5 colours: **332** elements (254 game state + 78 action masks).

## Reward

Agent reward = change in net worth (cash + harbour value + ship value + island scoring − loan penalties) between the current and next state. A terminal reward is computed at game end.

## Backward Compatibility

`transition()` accepts both flat-integer actions (from `ActionEncoder`) and multi-head arrays. Flat ints are converted to multi-head internally via `_flat_to_multihd()`, so existing code continues to work.

The `ActionEncoder` class also provides a `to_multi_head(action_idx)` method for explicit conversion.

## Running

```python
from container_rl.env.container import ContainerJaxEnv, ActionEncoder

env = ContainerJaxEnv(num_players=2, num_colors=5)
obs, info = env.reset()

# Legacy flat-int action (auto-converted)
encoder = ActionEncoder(2, 5)
action_idx = encoder.encode(0, {"color": 2})  # buy blue factory
obs, reward, term, trunc, info = env.step(action_idx)

# Multi-head action (preferred for RL training)
mh = encoder.to_multi_head(action_idx)
obs, reward, term, trunc, info = env.step(mh)
```

## Requirements

- Python 3.12+
- `jax>=0.10.0`
- `gymnasium>=1.3.0`
- `flax>=0.12.7`
- `pygame>=2.6.1` (for rendering)
