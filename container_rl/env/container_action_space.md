# Container Action Space v3 — No-Op + Per-Mode Masking

Each head has **no-op at index 0**. The mask determines which values the policy may select:

- **Parallel mode** (normal turn): no-op masked out on all 5 heads; all legal values shown.
- **Sequential mode** (shopping / produce continuation): irrelevant heads forced to no-op only; only the relevant head(s) are active.

Heads: `[action_type, opponent, colour, price_slot, purchase]`

Head sizes for 5 colours, **at any player count (2–5)**: `[12, 5, 6, 11, 32]`

## Player count

Head sizes do not depend on how many players are in the game, so one policy
can play any table size. The **opponent head is always 5 wide**: index 0 is
no-op and indices 1–4 are the other seats, counted clockwise from the acting
player. Seat *j* here is the same player as seat *j* in the observation.

In a game with fewer than 5 players the slots for seats that don't exist are
**masked in every mode** and can never be selected. In a 3-player game there
are two opponents, so indices 3 and 4 are always 0:

| Players | Opponent mask can be 1 at | Always 0 |
|---------|---------------------------|----------|
| 2 | 1 | 2, 3, 4 |
| 3 | 1, 2 | 3, 4 |
| 4 | 1, 2, 3 | 4 |
| 5 | 1, 2, 3, 4 | — |

The observation is laid out the same way: 5 identical seat blocks (acting
player first), then shared game state, then these masks. Absent seats are
filled with `NULL_OBS` (-2.0), and the game state starts with a `seat_present`
flag per seat. See `README.md` → *Observation Space*.

Purchase head (always 32, independent of player/colour count):
```
Index 0:  no-op
Index 1:  harbour $2  /  auction $1 bid  /  move_load "buy"
Index 2:  harbour $3  /  auction $2 bid
Index 3:  harbour $4  /  auction $3 bid
Index 4:  harbour $5  /  auction $4 bid
Index 5:  harbour $6  /  auction $5 bid
Indices 6–30:  auction bids $6–$30
Index 31:  STOP
```

---

## Action 0 — Buy Factory

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, buy_factory=1 | **1** (Buy Factory) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0, valid=1..nc | **3** (colour 2) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 31 (STOP) |

Turn advances immediately via `_advance_turn`.

---

## Action 1 — Buy Warehouse

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, buy_warehouse=1 | **2** (Buy Warehouse) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 31 (STOP) |

Turn advances immediately.

---

## Action 2 — Produce

Multiple steps: one initial entry, then colour+price per factory.

### Step 1 — Enter produce mode (parallel)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, produce=1 | **3** (Produce) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 31 (STOP) |

Agent pays $1 to union boss (if any factory can produce).  
`produce_active=1`, `produce_pending` = owned factory colours.  
Turn does NOT advance.

### Step 2+ — Recurrent produce (sequential, one step per factory)
| Head | Mask | Example pick |
|------|------|------|
| action_type | forced no-op (0) | 0 |
| opponent | forced no-op (0) | 0 |
| colour | no-op=0, pending factories=1 | **1** (colour 0) |
| price_slot | no-op=0, $1..$4 + leave idle=1 | **2** (slot 1 → $2) |
| purchase | forced no-op (0) | 0 |

Container produced → placed in factory store at chosen price.  
Factory marked processed in `produce_pending`.

If more factories pending → repeat Step 2+.  
If no factories left → `_finish_producing` → turn advances.

### Produce — "leave idle" example
| Head | Mask | Example pick |
|------|------|------|
| price_slot | no-op=0, $1..$4 + leave idle=1 | **5** (slot 4 → leave idle) |

No container produced. Factory still marked processed.  
Continue or finish as above.

---

## Action 3 — Buy from Factory Store (recurrent shopping)

### Step 1 — Select opponent (parallel)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, buy_factory_store=1 | **4** (Buy from Factory Store) |
| opponent | no-op=0, opponents with affordable stock=1 (within 1..np-1); absent seats np..4=0 | **1** (next seat clockwise) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 31 (STOP) |

Enters `shopping_active=1`, sets `shopping_target`. No purchase yet.  
Turn does NOT advance.

### Step 2+ — Recurrent purchases (sequential)
| Head | Mask | Example pick |
|------|------|------|
| action_type | forced no-op (0) | 0 |
| opponent | forced no-op (0) | 0 |
| colour | no-op=0, available colours from opponent=1 | **2** (colour 1) |
| price_slot | forced no-op (0) | 0 |
| purchase | no-op=0, harbour $2-$6 (1-5) + STOP (31)=1 | **2** (harbour $3) |

Environment auto-selects the cheapest source slot for the chosen colour.  
Harbour price is set per-container via the purchase head.  
If agent picks STOP (purchase=31): shopping ends, turn advances.

---

## Action 4 — Move to Harbour + Load (recurrent)

### Step 1 — Select opponent (parallel)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, move_load=1 | **5** (Move + Load) |
| opponent | no-op=0, opponents with affordable stock=1 (within 1..np-1); absent seats np..4=0 | **1** (next seat clockwise) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 31 (STOP) |

Enters `shopping_active=1`. Ship location set to target harbour.

### Step 2+ — Recurrent loads (sequential)
| Head | Mask | Example pick |
|------|------|------|
| action_type | forced no-op (0) | 0 |
| opponent | forced no-op (0) | 0 |
| colour | no-op=0, available colours=1 | **3** (colour 2) |
| price_slot | forced no-op (0) | 0 |
| purchase | no-op=0, 1=buy + 31=STOP | **1** (buy signal) |

Environment auto-selects the cheapest source slot for the chosen colour.  
Containers go to ship. No harbour price needed.  
If agent picks STOP (purchase=31): shopping ends, turn advances.

---

## Action 5 — Move to Open Sea

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, move_sea=1 | **6** (Move to Sea) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 31 (STOP) |

Ship location → `LOCATION_OPEN_SEA`. Turn advances.

---

## Action 6 — Move to Auction Island + Hold Auction

### Step 1 — Initiate auction (parallel, acting player)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, auction=1 | **7** (Auction) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 31 (STOP) |

### Step 2..N — Bidding (sequential, each non-seller player)
| Head | Mask | Example pick |
|------|------|------|
| action_type | only auction=1 | **7** (Auction) |
| opponent | bidder's **absolute** player index 0..np-1=1; absent seats=0 | **2** (player 2 bids) |
| colour | forced no-op (0) | 0 |
| price_slot | forced no-op (0) | 0 |
| purchase | $0 bid (index 0)=1, $1..cash=1 | **5** ($5 bid) |

Bid = `action[HEAD_PURCHASE]` directly (index 0 = $0 bid).  
Bidder = `action[HEAD_OPPONENT]` as an absolute player index — during an auction the opponent head is not seat-relative and index 0 means player 0, not no-op.

### Step N+1 — Seller decision (sequential)
| Head | Mask | Example pick |
|------|------|------|
| action_type | only auction=1 | **7** (Auction) |
| opponent | player index 0..np-1=1; absent seats=0 | **0** (the seller) |
| colour | forced no-op (0) | 0 |
| price_slot | forced no-op (0) | 0 |
| purchase | reject=0, accept=1 | **1** (Accept) |

---

## Action 7 — Pass

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, pass=1 | **8** (Pass) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 31 (STOP) |

Turn advances.

---

## Action 8 — Take Loan / Action 9 — Repay Loan

Single step, does NOT consume an action. Same masking pattern as Pass.

---

## Action 10 — Domestic Sale (variant)

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, domestic_sale=1 | **11** (Domestic Sale) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0, valid=store has container | **2** (colour 1) |
| price_slot | no-op=0, valid=slot has container | **5** (slot 4, $5) |
| purchase | no-op=0 | 31 (STOP) |

---

## Head Encoding Reference

| Head | Size (any players, 5c) | Index 0 | Indices meaning |
|------|-------------|---------|-----------------|
| action_type | 12 | no-op | 1=BuyFactory, 2=BuyWarehouse, 3=Produce, 4=BuyFromFactoryStore, 5=MoveLoad, 6=MoveSea, 7=Auction, 8=Pass, 9=TakeLoan, 10=RepayLoan, 11=DomesticSale |
| opponent | 5 | no-op | 1–4 = seats 1–4 clockwise from the acting player (seats past the last player masked); during an auction, 0..np-1 = absolute bidder index |
| colour | 6 | no-op | 1=colour 0, 2=colour 1, …, 5=colour 4 |
| price_slot | 11 | no-op | 1=$1, 2=$2, …, 10=$10 |
| purchase | 32 | no-op | 1–5=harbour $2–$6, 6–30=auction bids, 31=STOP |

All sub-head indices are offset by +1 from the old non-no-op encoding.
No-op is always index 0 and is either **masked out** (parallel mode) or **forced** (sequential mode for irrelevant heads).
The purchase head size is constant (32) and no longer depends on num_colors.
