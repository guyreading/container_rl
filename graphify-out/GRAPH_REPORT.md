# Graph Report - .  (2026-07-28)

## Corpus Check
- Corpus is ~49,532 words - fits in a single context window. You may not need a graph.

## Summary
- 932 nodes · 2044 edges · 51 communities (42 shown, 9 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 188 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Game Manager & Client
- Action Space Tests
- CLI & Entry Points
- Client TUI
- Action Space FuncEnv Tests
- Self-Play & Rankings
- SSH Auth & Handlers
- Container Functional Core
- Community & Governance
- Database & Persistence
- Action Space & Rules Docs
- Default Params & Config
- Turn Advance Tests
- Container Env Core
- Action Masks & Game State
- Action Encoder
- SSH TUI Menu
- Multi-head Action Tests
- SSH TUI Play
- SSH Client Connection
- Python Game Client
- JAX Environment
- Action Space Test Rationale
- Initial State Tests
- Observation & Reward Tests
- SSH TUI Root
- SSH TUI Messages
- SSH TUI Lobby
- Buy Factory Tests
- Net Worth Tests
- SSH TUI Register
- Action Mask Auction Tests
- SSH TUI Play Model
- Action Mask Produce Tests
- Produce Tests
- Rendering
- Multi-head Conversion
- Release Script
- Deploy Setup
- Documentation Index
- Issue Templates
- Issue Config
- Feature Request Template
- Container RL Package
- SSH Go Module

## God Nodes (most connected - your core abstractions)
1. `_make_state()` - 100 edges
2. `ContainerFunctional` - 84 edges
3. `EnvState` - 64 edges
4. `ActionEncoder` - 56 edges
5. `ContainerParams` - 53 edges
6. `_make_func_env()` - 44 edges
7. `_make_func_env()` - 40 edges
8. `TestActionMasksParallel` - 32 edges
9. `ContainerJaxEnv` - 30 edges
10. `Database` - 29 edges

## Surprising Connections (you probably didn't know these)
- `Container TUI Screenshot` --conceptually_related_to--> `Bubble Tea TUI`  [AMBIGUOUS]
  imgs/container_tui.png → container_rl/server/README.md
- `SHA-Pinned GitHub Actions` --semantically_similar_to--> `Dependabot 7-Day Cooldown Policy`  [INFERRED] [semantically similar]
  SECURITY.md → .github/dependabot.yml
- `TestActionMasksAuction` --uses--> `EnvState`  [INFERRED]
  tests/container_rl/env/test_action_space.py → container_rl/env/container.py
- `TestActionMasksParallel` --uses--> `EnvState`  [INFERRED]
  tests/container_rl/env/test_action_space.py → container_rl/env/container.py
- `TestActionMasksProduce` --uses--> `EnvState`  [INFERRED]
  tests/container_rl/env/test_action_space.py → container_rl/env/container.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Automated Security Analysis Tools** — github_workflows_codeql, github_workflows_zizmor, github_dependabot [INFERRED 0.85]
- **Contribution Lifecycle** — contributing, github_issue_templates_bug_report, github_issue_templates_feature_request, github_pull_request_template [INFERRED 0.85]
- **CI/CD Release Pipeline** — github_workflows_ci, github_workflows_docs, github_workflows_publish [INFERRED 0.85]
- **Action Masking Pipeline** — container_rl_env_readme_multihd_action_space, container_rl_env_readme_action_masks, container_rl_env_container_action_space_noop_masking, container_rl_env_container_action_space_parallel_mode, container_rl_env_container_action_space_sequential_mode [INFERRED 0.85]
- **SSH Server Architecture** — container_rl_server_readme_go_ssh_server, container_rl_server_readme_wish_ssh, container_rl_server_readme_bubbletea_tui, container_rl_server_readme_ssh_keys, container_rl_server_readme_auth_flow [INFERRED 0.95]

## Communities (51 total, 9 thin omitted)

### Community 0 - "Game Manager & Client"
Cohesion: 0.05
Nodes (35): TCP client for connecting to the game server., _describe_action(), GameManager, _opponent_name(), Any, Game lifecycle manager — create, join, start, and orchestrate turns., Join a game with a trusted player name (no password check)., Check if the lobby is full; if so, create the env and save initial state. (+27 more)

### Community 1 - "Action Space Tests"
Cohesion: 0.06
Nodes (34): _make_state(), P1 has affordable harbour stock and P0 has ship space. **Why**: need to give P1…, Ship in harbour — move to sea must be legal., Ship already at sea — move_sea is redundant but still legal (idempotent)., Ship at sea with cargo — auction must be legal., No cargo — auction not allowed., Pass is unconditionally legal., Loans < 2 — take_loan must be legal. (+26 more)

### Community 2 - "CLI & Entry Points"
Cohesion: 0.10
Nodes (50): command, _action_help(), _cname(), _compute_net_worth(), _cstyle(), _describe_action(), _describe_cargo(), _enter_raw_mode() (+42 more)

### Community 3 - "Client TUI"
Cohesion: 0.12
Nodes (48): _ch(), _cn(), _create_screen(), _cs(), _desc_cargo(), _drain_server(), _enter_raw(), _exit_raw() (+40 more)

### Community 4 - "Action Space FuncEnv Tests"
Cohesion: 0.08
Nodes (22): _make_func_env(), _make_params(), Enter produce mode, then process one factory at $2. **Why**: verifies the full…, Process two factories in the same produce batch. **Why**: the $1 union dues are…, Buy colour 1 from P1 at harbour price $3. **Why**: verifies the full two‑step…, Opponent has stock — shopping should be active after opponent selection., Harbour at capacity (1 warehouse → 1 container). **Why**: the…, Load colour 2 from P1's harbour onto an empty ship. **Why**: verifies the… (+14 more)

### Community 5 - "Self-Play & Rankings"
Cohesion: 0.09
Nodes (17): BaseCallback, Env, ndarray, RandomOpponent, rankings_from_net_worth(), Wraps a multi-player Container env for single-agent self-play training. The…, Compute final rankings (0 = winner) from net worth., Fallback opponent that samples uniformly from valid (unmasked) actions. (+9 more)

### Community 6 - "SSH Auth & Handlers"
Cohesion: 0.09
Nodes (24): AuthStatus, Handlers, main(), Fingerprint(), ParsePublicKey(), NewKeyStore(), findPython(), Session (+16 more)

### Community 7 - "Container Functional Core"
Cohesion: 0.09
Nodes (11): ContainerFunctional, Map opponent relative index to actual player, clockwise. opp_idx=0 is the…, Enter produce mode. On the first call each turn this initialises…, Process one factory: read colour + price_slot, produce container or leave idle.…, Select an opponent to buy from (step 1 of shopping). Sets ``shopping_active``…, Select an opponent to load from (step 1 of shopping). Sets ``shopping_active``…, Initiate an auction. Snapshots cargo, clears the ship, and enters recurrent…, Process one step of the recurrent auction. Bidding phase (auction_round == 0):… (+3 more)

### Community 8 - "Community & Governance"
Cohesion: 0.08
Nodes (32): Contributor Covenant 3.0 Code of Conduct, Contributor Covenant 3.0, Code of Conduct Enforcement Ladder, Contributing Guide, Dependabot Auto-Update Configuration, Dependabot 7-Day Cooldown Policy, Pull Request Template, AI Provenance Disclosure (+24 more)

### Community 9 - "Database & Persistence"
Cohesion: 0.11
Nodes (10): Connection, Database, _generate_code(), _hash_password(), SQLite persistence for games, players, and game state. Tables ------ players…, Find or create a player. Returns the player id., Find or create a player without password validation. Returns player id., Create a new game. Returns (game_id, game_code). (+2 more)

### Community 10 - "Action Space & Rules Docs"
Cohesion: 0.08
Nodes (30): No-Op + Per-Mode Masking (v3), Parallel Mode, Recurrent Produce, Sequential Mode, Auction Resolution, Container Board Game, Domestic Sale Variant, Interest Payment Errata (+22 more)

### Community 11 - "Default Params & Config"
Cohesion: 0.10
Nodes (26): ContainerParams, Parameters for the Container environment., Tests for the Container RL v3 action space. Covers all 11 action types under…, Verify masks during **produce continuation** (``produce_active = 1``). Only…, Verify masks during **auction** (``auction_active = 1``). Two sub‑modes: -…, Verify that masks change correctly when modes transition., Buying a factory costs ``(factories_owned + 1) * 3`` dollars. The agent cannot…, Buying a warehouse costs ``warehouses_owned + 3`` dollars. Max 5 warehouses.… (+18 more)

### Community 12 - "Turn Advance Tests"
Cohesion: 0.12
Nodes (5): _make_func_env(), TestAdvanceTurn, TestCheckGameEnd, TestHelpers, TestPayInterest

### Community 13 - "Container Env Core"
Cohesion: 0.10
Nodes (16): head_sizes(), mask_size(), Container game as a Gymnasium functional JAX environment. This implements the…, Total size of action mask vector appended to observation., Per-head category counts for MultiDiscrete action space. Each head includes a…, elo_update(), expected_score(), OpponentEntry (+8 more)

### Community 14 - "Action Masks & Game State"
Cohesion: 0.13
Nodes (14): ActType, Array, EnvState, Convert state to an ego-centric observation for the acting player. Per-player…, Check if game is over., Compute reward for the acting player (whoever ``current_player`` is) as their…, Game state transition implementing full Container rules. Accepts both legacy…, Clear shopping state and advance the turn. (+6 more)

### Community 15 - "Action Encoder"
Cohesion: 0.13
Nodes (8): ActionEncoder, Encode/decode between discrete action indices and their meaning., Decode action index into (action_type, params)., Encode action parameters to discrete index., Convert a flat action index to a multi-head action array. This provides…, parametrize, No two action types should overlap in discrete action space., TestActionEncoder

### Community 16 - "SSH TUI Menu"
Cohesion: 0.20
Nodes (10): debugLog(), Cmd, Model, Msg, NewCreateModel(), NewJoinModel(), NewMenuModel(), CreateModel (+2 more)

### Community 17 - "Multi-head Action Tests"
Cohesion: 0.08
Nodes (13): _build_multihd(), Build a v3 multi-head action array from *action_type* and optional *params*.…, Buy a second warehouse. **Why**: baseline purchase — cost deducted, warehouse…, Already at 5 warehouses — no change., $0 cash — warehouse stays at 1., Ship is in P1's harbour → moves to open sea. **Why**: verifies location changes…, Already at open sea — action is harmless (idempotent)., State is unchanged after a pass action. (+5 more)

### Community 18 - "SSH TUI Play"
Cohesion: 0.24
Nodes (6): gsInt(), gsInt2(), gsInt3(), NewPlayModel(), GameState, PlayModel

### Community 19 - "SSH Client Connection"
Cohesion: 0.14
Nodes (13): GameClient, Conn, Connect(), RawMessage, RawMessage, PackMessage(), ReadMessage(), UnmarshalPayload() (+5 more)

### Community 20 - "Python Game Client"
Cohesion: 0.15
Nodes (11): GameClient, Any, _drain_server(), _draw_list(), _enter_raw(), _exit_raw(), _key(), main() (+3 more)

### Community 21 - "JAX Environment"
Cohesion: 0.14
Nodes (11): ContainerJaxEnv, Standard Gymnasium ``Env`` wrapper around ``ContainerFunctional``. This class…, Top-level package for Container RL Env., EzPickle, FunctionalJaxEnv, Run a full game with JIT-disabled environment until termination., Verify the game terminates when 2 colors are exhausted., Verify the JIT-compiled environment works end-to-end. (+3 more)

### Community 22 - "Action Space Test Rationale"
Cohesion: 0.16
Nodes (10): Verify masks during **shopping continuation** (``shopping_active = 1``). Two…, Create a state mid‑shopping with given target opponent., Action type head MUST be forced to no‑op only during shopping. **Why**: the…, Opponent head forced to no‑op during shopping. **Why**: the opponent was locked…, Price-slot head forced to no‑op during shopping. **Why**: the cheapest source…, Colour head shows available colours from the target opponent. **Why**: P1…, Colour head shows available colours from target's harbour. **Why**: P1 harbour…, During factory shopping purchase head shows $2-$6 + STOP. **Why**: indices 1-5… (+2 more)

### Community 24 - "Observation & Reward Tests"
Cohesion: 0.24
Nodes (4): _make_params(), Unit and integration tests for the Container RL environment., TestObservationTerminalReward, TestTransition

### Community 25 - "SSH TUI Root"
Cohesion: 0.19
Nodes (9): Cmd, Model, Msg, Session, NewRootModel(), GameClient, ProgramOption, RootModel (+1 more)

### Community 26 - "SSH TUI Messages"
Cohesion: 0.20
Nodes (11): actionResultMsg, errMsg, gameCreatedMsg, GameInfo, gameJoinedMsg, gameListMsg, gameStartedMsg, lobbyUpdateMsg (+3 more)

### Community 27 - "SSH TUI Lobby"
Cohesion: 0.33
Nodes (6): Cmd, Model, Msg, NewLobbyModel(), LobbyModel, lobbyTickMsg

### Community 28 - "Buy Factory Tests"
Cohesion: 0.18
Nodes (7): ndarray, Convert an old-style (action_type, rel_offset) to a v3 multi-head array.…, Buy a factory of colour 2 (not yet owned). **Why**: verify the basic purchase…, Attempt to buy colour 0 — already owned. **Why**: the handler must be…, All 5 colours already owned — purchase should be rejected silently., Only $1 cash — cannot afford a $3+ factory., _rel_to_multihd()

### Community 29 - "Net Worth Tests"
Cohesion: 0.18
Nodes (3): 10/5 color = $10 when you have at least one of every color., 10/5 color = $5 when you don't have all colors., TestNetWorth

### Community 30 - "SSH TUI Register"
Cohesion: 0.33
Nodes (5): Cmd, Model, Msg, NewRegisterModel(), RegisterModel

### Community 31 - "Action Mask Auction Tests"
Cohesion: 0.20
Nodes (5): State mid‑auction. Default: P1 is bidding, P0 is seller., Colour and price_slot heads forced to no‑op during auction. Opponent head is…, During bidding, purchase head shows $0 bid + $1..cash bids. **Why**: index 0 =…, Player has only $5 — bid range must be $0..$5 only., Seller (P0) sees only reject (index 0) or accept (index 1). **Why**: the seller…

### Community 32 - "SSH TUI Play Model"
Cohesion: 0.36
Nodes (4): jsonUnmarshal(), Cmd, Model, Msg

### Community 33 - "Action Mask Produce Tests"
Cohesion: 0.25
Nodes (4): State mid‑produce with colour 0 and colour 2 pending. P0 owns colours 0 and 2;…, Action type, opponent, and purchase heads forced to no‑op only. **Why**: during…, Only pending factories (colours 0, 2) are selectable. **Why**:…, Only $1-$4 (indices 1-4) + leave idle (index 5) are valid. **Why**:…

### Community 34 - "Produce Tests"
Cohesion: 0.32
Nodes (5): Produce is a recurrent action with two kinds of step: 1. **Enter** (parallel) —…, Helper: enter produce mode, optionally process one factory., Player 1 takes the produce action — pays player 0 (right). **Why**: the union…, Factory store at capacity (2 containers for 1 factory). **Why**:…, TestProduce

### Community 35 - "Rendering"
Cohesion: 0.33
Nodes (3): ndarray, RenderStateType, StateType

### Community 36 - "Multi-head Conversion"
Cohesion: 0.50
Nodes (3): num_heads(), Convert a legacy flat action index to a multi-head action array. For shopping…, Total number of action heads (always 5).

## Ambiguous Edges - Review These
- `Bubble Tea TUI` → `Container TUI Screenshot`  [AMBIGUOUS]
  imgs/container_tui.png · relation: conceptually_related_to

## Knowledge Gaps
- **37 isolated node(s):** `github.com/guyreading/container-rl-ssh`, `AuthStatus`, `errMsg`, `gameCreatedMsg`, `gameJoinedMsg` (+32 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Bubble Tea TUI` and `Container TUI Screenshot`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `EnvState` connect `Action Masks & Game State` to `Game Manager & Client`, `Action Space Tests`, `CLI & Entry Points`, `Client TUI`, `Action Space FuncEnv Tests`, `Produce Tests`, `Container Functional Core`, `Default Params & Config`, `Turn Advance Tests`, `Container Env Core`, `Action Encoder`, `JAX Environment`, `Action Space Test Rationale`, `Initial State Tests`, `Observation & Reward Tests`, `Net Worth Tests`?**
  _High betweenness centrality (0.188) - this node is a cross-community bridge._
- **Why does `ContainerFunctional` connect `Container Functional Core` to `Action Space Tests`, `Produce Tests`, `Rendering`, `Multi-head Conversion`, `Self-Play & Rankings`, `Action Space FuncEnv Tests`, `Default Params & Config`, `Turn Advance Tests`, `Container Env Core`, `Action Masks & Game State`, `Action Encoder`, `JAX Environment`, `Action Space Test Rationale`, `Initial State Tests`, `Observation & Reward Tests`, `Net Worth Tests`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `_make_state()` connect `Action Space Tests` to `Action Mask Produce Tests`, `Produce Tests`, `Action Space FuncEnv Tests`, `Default Params & Config`, `Turn Advance Tests`, `Action Masks & Game State`, `Multi-head Action Tests`, `Action Space Test Rationale`, `Observation & Reward Tests`, `Buy Factory Tests`, `Net Worth Tests`, `Action Mask Auction Tests`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Are the 32 inferred relationships involving `_make_state()` (e.g. with `.test_auction_ends_turn_immediately()` and `.test_first_action_does_not_end_turn()`) actually correct?**
  _`_make_state()` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `ContainerFunctional` (e.g. with `OpponentEntry` and `OpponentPool`) actually correct?**
  _`ContainerFunctional` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `EnvState` (e.g. with `GameManager` and `TestActionMasksAuction`) actually correct?**
  _`EnvState` has 27 INFERRED edges - model-reasoned connections that need verification._