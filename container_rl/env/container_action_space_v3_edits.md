# Container Action Space v3 — No-Op + Per-Mode Masking

Each head has **no-op at index 0**. The mask determines which values the policy may select:

- **Parallel mode** (normal turn): no-op masked out on all 5 heads; all legal values shown.
- **Sequential mode** (shopping / produce continuation): irrelevant heads forced to no-op only; only the relevant head(s) are active.

Heads: `[action_type, opponent, colour, price_slot, purchase]`

Head sizes for 4 players, 5 colours: `[12, 4, 6, 11, 52]`

---

## Action 0 — Buy Factory

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, buy_factory=1 | **1** (Buy Factory) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0, valid=1..nc | **3** (colour 2) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 8 (no-op, forced through multi-head defaults) |

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
| purchase | no-op=0 | 51 (STOP) |

Turn advances immediately.

---

## Action 2 — Produce

Multiple sub-steps: one initial entry, then colour-price per factory.

### Step 1 — Enter produce mode (parallel)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, produce=1 | **3** (Produce) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 51 (STOP) |

Agent pays $1 to union boss (if any factory can produce).  
`produce_active=1`, `produce_sub_phase=0`, `produce_pending` = owned factory colours.  
Turn does NOT advance.

### Step 2+ — Recurrent Pick colour & price (sequential)
| Head | Mask | Example pick |
|------|------|------|
| action_type | forced no-op (0) | 0 |
| opponent | forced no-op (0) | 0 |
| colour | no-op=0, pending factories=1 | **1** (colour 0 — first owned factory) |
| price_slot | no-op=0, $1..$4 + leave idle=1 | **2** (price slot 1 → $2) |
| purchase | forced no-op (0) | 0 |

`produce_sub_phase=1`, `produce_current_color=0`.  
Turn does NOT advance.

Container produced → placed in factory store at chosen price.  
Factory marked processed in `produce_pending`.

If more factories pending: `produce_sub_phase=0` → repeats from Step 2.  
If no factories left: `_finish_producing` → turn advances.

### Produce — "leave idle" example (no container produced)
| Head | Mask | Example pick |
|------|------|------|
| price_slot | no-op=0, $1..$4 + leave idle=1 | **5** (slot 4 → leave idle) |

No container produced. Factory still marked processed.  
Continue or finish as above.

---

## Action 3 — Buy from Factory Store (recurrent shopping)

### Step 1 — Initialise purchase & choose opponent (parallel)
Agent picks action type and opponent at first.

| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, buy_factory_store=1 | **4** (Buy from Factory Store) |
| opponent | no-op=0, valid opponents=1..np-1 | **1** (opponent 0, first clockwise) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 51 (STOP) |



### Step 2+ — Recurrent purchases (sequential)
Agent picks the container colour and the price to set it in the harbour store.
| Head | Mask | Example pick |
|------|------|------|
| action_type | forced no-op (0) | 0 |
| opponent | forced no-op (0) | 0 |
| colour | no-op=0, buy colour=1 | **1** (colour 0 — first owned factory) |
| price_slot | forced no-op (0) | 0 |
| purchase | no-op=0, valid(affordable+space)+STOP=1 | **26** (colour 2, source slot 5) |

When a colour is selected, the cheapest colour available is automatically chosen by the game environment. In this way, we don't need to use the price_slot head. If a colour is not available, it is action masked out. We set the price of the colour in the same step. The available prices are within the valid harbour prices.

One container purchased. If more purchases possible:
`shopping_active=1`, `shopping_harbour_price` saved. Turn does NOT advance.

If no more possible (or STOP selected): turn advances.

Harbour price **locked** from Step 1 via `shopping_harbour_price`.  
If agent picks STOP (`purchase >= nc×10+1` = index 51): shopping ends, turn advances.

---

## Action 4 — Move to Harbour + Load (recurrent)

Same structure as Action 3, but containers go to the ship instead of harbour.

### Step 1 — Initialise purchase & choose opponent (parallel)
Agent picks action type and opponent at first.

| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, buy_factory_store=1 | **4** (Buy from Factory Store) |
| opponent | no-op=0, valid opponents=1..np-1 | **1** (opponent 0, first clockwise) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 51 (STOP) |


### Step 2+ — Additional loads (sequential)
| Head | Mask | Example pick |
|------|------|------|
| action_type | forced no-op (0) | 0 |
| opponent | forced no-op (0) | 0 |
| colour | no-op=0, buy colour=1 | **1** (colour 0 — first owned factory) |
| price_slot | forced no-op (0) | 0 |
| purchase | no-op=0, valid+STOP=1 | **41** (colour 4, source slot 0, $1) |

`shopping_active=1` if more possible. Ship location set to target harbour.

When a colour is selected, the cheapest colour available is automatically chosen by the game environment. In this way, we don't need to use the price_slot head. If a colour is not available, it is action masked out. We don't need to set the price when ship loads (containers go to ship, not harbour store). 

---

## Action 5 — Move to Open Sea

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, move_sea=1 | **6** (Move to Sea) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 51 (STOP) |

Ship location → `LOCATION_OPEN_SEA`. Turn advances.

---

## Action 6 — Move to Auction Island + Hold Auction

Multiple sub-steps: ship moves, then each player bids, then seller decides.

### Step 1 — Initiate auction (parallel, acting player)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, auction=1 | **7** (Auction) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 51 (STOP) |

Cargo snapshotted from ship. Ship cleared, moved to auction island.  
`auction_active=1`, current player advances to first bidder. Turn does NOT advance.

### Step 2..N — Bidding (sequential, each non-seller player)
| Head | Mask | Example pick |
|------|------|------|
| action_type | only auction=1 | **7** (Auction) |
| opponent | forced no-op (0) | 0 |
| colour | forced no-op (0) | 0 |
| price_slot | forced no-op (0) | 0 |
| purchase | $0 bid (index 0)=1, $1..cash=1 | **5** ($5 bid) |

Bid = `action[HEAD_PURCHASE]` directly (index 0 = $0 bid).  
After all bids collected, current player → seller.

### Step N+1 — Seller decision (sequential)
| Head | Mask | Example pick |
|------|------|------|
| action_type | only auction=1 | **7** (Auction) |
| opponent | forced no-op (0) | 0 |
| colour | forced no-op (0) | 0 |
| price_slot | forced no-op (0) | 0 |
| purchase | reject=0, accept=1 | **1** (Accept) |

Auction resolves: goods deposited, cash exchanged, `auction_active=0`.  
Turn advances (auction always ends the turn).

---

## Action 7 — Pass

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, pass=1 | **8** (Pass) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 51 (STOP) |

No state change. Turn advances.

---

## Action 8 — Take Loan

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, take_loan=1 | **9** (Take Loan) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 51 (STOP) |

Cash +$10, loans +1. Does NOT consume an action.

---

## Action 9 — Repay Loan

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, repay_loan=1 | **10** (Repay Loan) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0 | 0 (no-op, forced) |
| price_slot | no-op=0 | 0 (no-op, forced) |
| purchase | no-op=0 | 51 (STOP) |

Cash -$10, loans -1. Does NOT consume an action.

---

## Action 10 — Domestic Sale (variant)

### Step 1 (parallel, single step)
| Head | Mask | Example pick |
|------|------|------|
| action_type | no-op=0, domestic_sale=1 | **11** (Domestic Sale) |
| opponent | no-op=0 | 0 (no-op, forced) |
| colour | no-op=0, valid=store has container | **2** (colour 1) |
| price_slot | no-op=0, valid=slot has container | **5** (slot 4, $5) |
| purchase | no-op=0 | 51 (STOP) |

One container returned to supply, player gets $2. Turn advances.

---

## Head Encoding Reference

| Head | Size (2p,5c) | Index 0 | Indices meaning |
|------|-------------|---------|-----------------|
| action_type | 12 | no-op | 1=BuyFactory, 2=BuyWarehouse, 3=Produce, 4=BuyFromFactoryStore, 5=MoveLoad, 6=MoveSea, 7=Auction, 8=Pass, 9=TakeLoan, 10=RepayLoan, 11=DomesticSale |
| opponent | 2 | no-op | 1=first clockwise opponent |
| colour | 6 | no-op | 1=colour 0, 2=colour 1, …, 5=colour 4 |
| price_slot | 11 | no-op | 1=$1, 2=$2, …, 10=$10 |
| purchase | 52 | no-op | 1..50 = `1 + colour×10 + source_slot`, 51 = STOP |

All sub-head indices are offset by +1 from the old non-no-op encoding.
No-op is always index 0 and is either **masked out** (parallel mode) or **forced** (sequential mode for irrelevant heads).
