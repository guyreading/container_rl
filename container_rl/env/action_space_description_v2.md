# Action Space v2 — Hybrid Multi-Head with No-Op Masking

This builds on the no-op idea from v1, but avoids the timestep explosion of
fully-sequential selection.  Heads are selected **in parallel** for initial
actions, switching to **sequential no-op masking only during recurrent
continuation** (shopping, produce batch).  The purchase head is kept because
two prices must be specified when buying from the factory store.

---

## Head Layout (4 player, 5 colour)

| Head | Name | Size | Description |
|------|------|------|-------------|
| 0 | **Action Type** | 12 | 11 action types + no-op |
| 1 | **Opponent** | 4 | 3 opponents clockwise + no-op |
| 2 | **Colour** | 6 | 5 colours + no-op |
| 3 | **Price Slot** | 11 | 10 price slots ($1–$10) + no-op |
| 4 | **Purchase** | `nc × PRICE_SLOTS + 2` | `1 + color × PRICE_SLOTS + source_price`, or STOP at final index; no-op at 0 |

**no-op is always index 0** on every head.

Prices are 1-indexed relative to the slots: slot 0 → $1, slot 9 → $10.
For produce the valid price range is $1–$4 (slots 0–3), with slot 4 meaning
"leave this factory idle".

---

## Two Operation Modes

### 1. Parallel mode — initial action selection

All heads are active.  The agent picks the action type AND all its parameters
in a single step.  no-op is **masked out** on every head.

```
Step 1 (new turn):
[~~0~~ 1 2 3 4 5 6 7 8 9 10 11]   # action type   (no-op masked)
[~~0~~ 1 2 3]                      # opponent       (no-op masked)
[~~0~~ 1 2 3 4 5]                  # colour         (no-op masked)
[~~0~~ 1 2 3 4 5 6 7 8 9 10]      # price slot     (no-op masked)
[~~0~~ 1 2 3 … STOP]              # purchase       (no-op masked)
```

**Example — Buy Factory:**
Action type = 1, colour = 2.  Opponent / price_slot / purchase are
don't-care — the action handler ignores them.  Turn advances immediately.

**Example — Buy from Factory Store (single container):**
Action type = 3, opponent = 1, purchase = `1 + color × PRICE_SLOTS + source_slot`,
price_slot = harbour destination price ($2–$6, slots 1–5).
All in one step.  If more purchases are possible, `shopping_active` is set.

### 2. Sequential mode — shopping / produce continuation

When the environment enters a recurrent phase (set by `shopping_active` or
`produce_active` in the state), the action type and other irrelevant heads
are **forced to no-op**.  Only the heads needed for the current sub-decision
are active.

#### 2a. Shopping continuation (buying more containers)

After a `BUY_FROM_FACTORY_STORE` or `MOVE_LOAD` action, if more purchases
are possible, the agent stays in shopping mode:

```
Step N+1 (mid-shopping):
[0 ~~1 2 3 4 5 6 7 8 9 10 11~~]   # action type → forced no-op
[0 ~~1 2 3~~]                      # opponent      → forced no-op
[0 ~~1 2 3 4 5~~]                  # colour        → forced no-op
[0 ~~1 2 3 4 5 6 7 8 9 10~~]      # price slot    → forced no-op
[~~0~~ 1 2 3 … STOP]              # purchase       → active (which container, or STOP)
```

The action type and target opponent are locked in from the initial action.
The agent only decides WHICH container to buy next via the purchase head.

**Harbour price during shopping:** the price_slot from the INITIAL action
applies to ALL containers bought in that shopping session.  If the agent
wants different prices for different containers it must end shopping and
start a new `BUY_FROM_FACTORY_STORE` action.  This matches the rule:
"Each time you perform this action you may also reorganise all the containers
in your Harbour Store, repricing them."

If the agent selects STOP (purchase head's final index) the shopping phase
ends and the turn advances.

#### 2b. Produce batch continuation (one factory at a time)

The produce action works in two sub-phases per factory: colour, then price.

```
Step N+1 — pick factory colour:
[0 ~~1 2 3 4 5 6 7 8 9 10 11~~]   # action type → forced no-op
[0 ~~1 2 3~~]                      # opponent      → forced no-op
[~~0~~ 1 2 ~~3 4 5~~]              # colour        → only pending factories
[0 ~~1 2 3 4 5 6 7 8 9 10~~]      # price slot    → forced no-op
[0 ~~1 2 3 … STOP~~]              # purchase      → forced no-op
```

```
Step N+2 — pick price:
[0 ~~1 2 3 4 5 6 7 8 9 10 11~~]   # action type → forced no-op
[0 ~~1 2 3~~]                      # opponent      → forced no-op
[0 ~~1 2 3 4 5~~]                  # colour        → forced no-op
[~~0~~ 1 2 3 4 ~~5 6 7 8 9 10~~]  # price slot    → $1–$4 + leave idle
[0 ~~1 2 3 … STOP~~]              # purchase      → forced no-op
```

After each factory is processed, if more factories remain pending the loop
continues with the next colour-selection step.  When no factories remain,
the batch ends and the turn advances.

---

## Why This Beats Fully-Sequential

1. **Fewer steps per turn.**  A normal buy-factory is 1 step, not 2.
   Buy-from-factory-store (1 container) is 1 step, not 4.  Shopping
   continuation for additional containers is 1 step each (same as now).
   Produce batch adds 2 steps per factory (colour + price) instead of 1,
   but produce is the minority case — most turns don't produce.

2. **Denser reward signal.**  Because the state changes in every step
   (cash, stores) the agent gets immediate reward feedback on most steps.
   Fully-sequential has many zero-reward "selection" steps.

3. **No protocol learning overhead.**  The agent doesn't need to learn the
   "grammar" of valid sub-action sequences — the parallel multi-head
   selection makes the intent explicit in one step.

4. **The no-op abstraction still pays off during continuation.**  During
   shopping/produce, forcing irrelevant heads to no-op makes the masks
   trivially correct and prevents the agent from exploring dead dimensions.

---

## What Changes from Current Code

| Current | v2 |
|---------|-----|
| 5 heads, no no-ops | 5 heads, no-op at index 0 of each |
| Purchase STOP at `nc × PRICE_SLOTS` | Purchase STOP at index 1 + `nc × PRICE_SLOTS` (no-op shifts by 1) |
| Masks computed as union across all action contexts | Masks computed per-mode: parallel shows all; sequential shows only relevant heads |
| Shopping `price_slot` not masked | Shopping `price_slot` forced to no-op (locked from initial action) |
| `shopping_active` tracks phase | Same, plus sequential masking per sub-phase |
| `produce_active` / `produce_pending` | Same, plus 2 sub-phases per factory (colour then price) |
| Head sizes: [11, np-1, nc, 10, nc×10+1] | Head sizes: [12, np, nc+1, 11, nc×10+2] (all +1 for no-op) |
