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

## Action Space — Multi-Head Design (5 heads)

The action space uses a **multi-head** architecture inspired by [settlers-rl](https://settlers-rl.github.io). Instead of a single flat discrete index, the action is a fixed-size `int32` array of 5 elements, where each element (head) handles one sub-decision. The overall action log-probability for PPO training is the sum of the log-probs from each relevant head.

### Why Multi-Head?

- **Generalisation** — Feedback from one action choice (e.g. buying blue containers) can reinforce the *colour* head independently of the *price* or *opponent* heads, improving sample efficiency.
- **Action masking** — The environment provides per-head binary masks; invalid options are zeroed out before softmax so the agent never wastes time on illegal actions.
- **Recurrent shopping (Option B)** — Actions 3 and 4 can buy **multiple containers in a single action** (as the real rules allow). After the first purchase the environment sets `shopping_active = True`; the training loop calls the policy again for each subsequent purchase until STOP is selected or constraints are exhausted. Log-probabilities from each sub-step are summed — exactly matching the PPO formula from settlers-rl.

### Head Layout

The action tensor is an `int32` array of shape `(5,)`:

| Head index | Name | Size | Used by | Description |
|---|---|---|---|---|
| 0 | **Action Type** | 11 | all | One of the 11 action types |
| 1 | **Opponent** | `n_players − 1` | BuyFromFactoryStore (3), MoveLoad (4) | Which opponent to target (clockwise from acting player) |
| 2 | **Color** | `n_colors` | BuyFactory (0), DomesticSale (10) | Which container colour |
| 3 | **Price Slot** | `PRICE_SLOTS` (10) | DomesticSale (10) | Which price slot ($1–$10) to sell from |
| 4 | **Purchase** | `n_colors × PRICE_SLOTS + 1` | BuyFromFactoryStore (3), MoveLoad (4) | Single purchase: `color × PRICE_SLOTS + price_slot`, or STOP at index `n_colors × PRICE_SLOTS` |

### Which Heads Are Active Per Action

| Action | Action Type | Opponent | Color | Price Slot | Purchase |
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

For actions **3** (Buy from Factory Store) and **4** (Move to Harbour + Load):

1. The first `env.step(action)` executes a single purchase. If the purchase was valid and more purchases are possible, the environment sets `state.shopping_active = 1` and does **not** advance the turn.
2. The training loop detects `shopping_active` and calls the policy again, passing the updated observation.
3. Each subsequent `env.step(action)` executes one more purchase. When the agent selects STOP or constraints prevent further purchases, `shopping_active` is cleared and the turn advances (consuming one action total for the entire shopping sequence).
4. The total log-prob for PPO = sum of log-probs from the first step plus each shopping continuation step.

```python
# Pseudo-code for the training loop
action, log_prob = policy.sample(obs, masks)
obs, reward, _, _, info = env.step(action)
total_lp = log_prob

while env.state.shopping_active:
    masks = env.func_env._action_masks(env.state, env.func_env.params)
    sub_action, sub_lp = policy.sample(obs, masks)
    obs, sub_r, _, _, _ = env.step(sub_action)
    total_lp += sub_lp
    reward += sub_r
```

### Action Masks

`ContainerFunctional._action_masks(state, params)` returns per-head binary vectors:

| Key | Shape | Meaning |
|---|---|---|
| `action_type` | (11,) | 1 = action type is currently legal |
| `opponent` | (n_players−1,) | 1 = opponent has purchasable stock |
| `color` | (n_colors,) | 1 = colour is a valid choice |
| `price_slot` | (PRICE_SLOTS,) | 1 = own store has containers at that price |
| `purchase` | (n_colors×10+1,) | 1 = (color, slot) combo is affordable & in stock; STOP always valid |

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
         + 3                         # shopping_active, shopping_action_type, shopping_target
         + mask_size                 # action_type(11) + opponent(np-1) + color(nc) + price_slot(10) + purchase(nc*10+1)
```

For 2 players, 5 colours: **335** elements (254 game state + 3 shopping + 78 action masks).

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
