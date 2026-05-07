"""TUI for playing the Container board game via the terminal.

Features:
- Terminal-resident display (no scrolling)
- Player boxes in a horizontal row
- State history with left/right arrow navigation
- Keyboard-controlled action selection
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from typing import TYPE_CHECKING

import jax
import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from container_rl.env.container import (
    ACTION_BUY_FACTORY,
    ACTION_BUY_FROM_FACTORY_STORE,
    ACTION_BUY_WAREHOUSE,
    ACTION_DOMESTIC_SALE,
    ACTION_MOVE_AUCTION,
    ACTION_MOVE_LOAD,
    ACTION_MOVE_SEA,
    ACTION_PASS,
    ACTION_PRODUCE,
    ACTION_REPAY_LOAN,
    ACTION_TAKE_LOAN,
    FACTORY_STORAGE_MULTIPLIER,
    LEAVE_IDLE,
    LOCATION_AUCTION_ISLAND,
    LOCATION_HARBOUR_OFFSET,
    LOCATION_OPEN_SEA,
    MAX_FACTORIES_PER_PLAYER,
    PRODUCE_CHOICES,
    PRICE_SLOTS,
    SHIP_CAPACITY,
    ActionEncoder,
    ContainerJaxEnv,
    EnvState,
)

if TYPE_CHECKING:
    pass

jax.config.update("jax_disable_jit", True)

app = typer.Typer()
console = Console()

# ── colour palette ───────────────────────────────────────────────────────────

COLOR_NAMES = ["Red", "Blue", "Green", "Yellow", "Purple"]
COLOR_STYLES = ["red", "blue", "green", "yellow", "magenta"]
COLOR_EMOJI = ["🔴", "🔵", "🟢", "🟡", "🟣"]


def _cname(idx: int, num_colors: int) -> str:
    return COLOR_NAMES[idx] if idx < len(COLOR_NAMES) else f"Color {idx}"


def _cstyle(idx: int) -> str:
    return COLOR_STYLES[idx] if idx < len(COLOR_STYLES) else "white"


# ── helpers ──────────────────────────────────────────────────────────────────


def _opponent_menu_indices(current_player: int, num_players: int) -> list[int]:
    return [p for p in range(num_players) if p != current_player]


# ── state rendering ──────────────────────────────────────────────────────────


def _render_store_compact(store, player: int, num_colors: int) -> str:
    lines = []
    for c in range(num_colors):
        entries = []
        for s in range(PRICE_SLOTS):
            cnt = int(store[player, c, s])
            if cnt > 0:
                entries.append(f"{cnt}×${s + 1}")
        if entries:
            lines.append(f"  [{_cstyle(c)}]{_cname(c, num_colors)}[/{_cstyle(c)}]: {', '.join(entries)}")
    if not lines:
        return "[dim](empty)[/dim]"
    return "\n".join(lines)


def _render_store_table(store, player: int, num_colors: int) -> Table:
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1), show_edge=False)
    table.add_column("Col", style="bold", width=5)
    for s in range(PRICE_SLOTS):
        table.add_column(f"${s + 1}", justify="right", width=3)
    table.add_column("∑", justify="right", width=3)

    for c in range(num_colors):
        counts = [int(store[player, c, s]) for s in range(PRICE_SLOTS)]
        total = sum(counts)
        if total == 0:
            continue
        row = [f"[{_cstyle(c)}]{_cname(c, num_colors)}[/{_cstyle(c)}]"]
        row += [str(cnt) if cnt > 0 else "·" for cnt in counts]
        row.append(str(total))
        table.add_row(*row)

    if len(table.rows) == 0:
        table.add_row("[dim](empty)[/dim]", *([""] * (PRICE_SLOTS + 1)))
    return table


def _render_island(island_store, player: int, num_colors: int) -> str:
    parts = []
    for c in range(num_colors):
        cnt = int(island_store[player, c])
        if cnt > 0:
            parts.append(f"[{_cstyle(c)}]{cnt}×{_cname(c, num_colors)}[/{_cstyle(c)}]")
    return "  ".join(parts) if parts else "[dim](empty)[/dim]"


def _render_ship(state: EnvState, player: int) -> str:
    row = state.ship_contents[player]
    parts = []
    for i in range(SHIP_CAPACITY):
        c = int(row[i])
        parts.append(f"[{_cstyle(c - 1)}]{'■'}[/{_cstyle(c - 1)}]" if c > 0 else "·")

    loc = int(state.ship_location[player])
    if loc == LOCATION_OPEN_SEA:
        loc_str = "[cyan]Open Sea[/cyan]"
    elif loc == LOCATION_AUCTION_ISLAND:
        loc_str = "[yellow]Auction Isl.[/yellow]"
    elif loc >= LOCATION_HARBOUR_OFFSET:
        loc_str = f"[green]P{loc - LOCATION_HARBOUR_OFFSET + 1}'s Harbour[/green]"
    else:
        loc_str = str(loc)
    return f"{' '.join(parts)}  @ {loc_str}"


def _compute_net_worth(state: EnvState, player: int, num_colors: int) -> int:
    cash = int(state.cash[player])
    loans_penalty = int(state.loans[player]) * 11
    harbour_val = int(jax.numpy.sum(state.harbour_store[player])) * 2
    ship_val = int(jax.numpy.sum(state.ship_contents[player] > 0)) * 3
    secret = int(state.secret_value_color[player])
    island = state.island_store[player]
    has_all = int(jax.numpy.all(island > 0))
    island_val = 0
    for c in range(num_colors):
        cnt = int(island[c])
        if cnt > 0:
            island_val += cnt * (10 if (c == secret and has_all) else 5 if c == secret else 2)
    return cash + harbour_val + ship_val + island_val - loans_penalty


def _player_card(state: EnvState, player: int, nc: int, is_current: bool, width: int = 28) -> Text:
    """Render a compact one-player card as a Text object."""
    cash = int(state.cash[player])
    loans = int(state.loans[player])
    wh = int(state.warehouse_count[player])
    secret = int(state.secret_value_color[player])
    nw = _compute_net_worth(state, player, nc)

    factories = []
    for c in range(nc):
        if int(state.factory_colors[player, c]):
            factories.append(f"[{_cstyle(c)}]{_cname(c, nc)}[/{_cstyle(c)}]")
    fac_str = ", ".join(factories) if factories else "[dim]none[/dim]"

    current_badge = " [bold white on green]◄[/bold white on green]" if is_current else ""
    header = f"[bold]P{player + 1}{current_badge}[/bold]  ${nw}"
    line = "─" * max(0, width - len(header) + 10)

    out = Text.from_markup(header + "\n")
    out.append(line + "\n")
    out.append_text(Text.from_markup(f"  💵 ${cash}  🏦 {loans} loans  🏭 {wh} wh\n"))
    out.append_text(Text.from_markup(f"  🤫 [{_cstyle(secret)}]{_cname(secret, nc)}[/{_cstyle(secret)}]\n"))
    out.append_text(Text.from_markup(f"  Factories: {fac_str}\n"))
    out.append("\n")
    out.append_text(Text.from_markup("  [bold]Factory Store:[/bold]\n"))
    out.append_text(Text.from_markup(_render_store_compact(state.factory_store, player, nc) + "\n"))
    out.append_text(Text.from_markup("  [bold]Harbour Store:[/bold]\n"))
    out.append_text(Text.from_markup(_render_store_compact(state.harbour_store, player, nc) + "\n"))
    out.append_text(Text.from_markup(f"  🏝️ {_render_island(state.island_store, player, nc)}\n"))
    out.append_text(Text.from_markup(f"  🚢 {_render_ship(state, player)}"))
    return out


def _supply_bar(state, nc: int) -> Text:
    parts = []
    exhausted = 0
    for c in range(nc):
        cnt = int(state.container_supply[c])
        if cnt <= 0:
            parts.append(f"[{_cstyle(c)}]{_cname(c, nc)}[/{_cstyle(c)}]: [red]0[/red]")
            exhausted += 1
        else:
            bar = "█" * min(cnt, 10)
            parts.append(f"[{_cstyle(c)}]{_cname(c, nc)}[/{_cstyle(c)}]: {bar} {cnt}")
    text = "  │  ".join(parts)
    text += f"  │  [bold]Exhausted: {exhausted}/2[/bold]"
    return Text.from_markup(text)


def _action_help(factory_cost: int = 0) -> Text:
    fac_str = f"BuyFactory (${factory_cost})" if factory_cost > 0 else "BuyFactory"
    return Text.from_markup(
        f" [1]{fac_str}  [2]BuyWarehouse  [3]Produce  [4]BuyFromFactory  [5]LoadShip\n"
        " [6]MoveToSea   [7]Auction    [p]Pass    [l]TakeLoan        [r]RepayLoan  [d]DomesticSale\n"
        " [←→] history  [q]uit"
    )


def _render_frame(
    state: EnvState,
    nc: int,
    num_players: int,
    hist_msg: str = "",
    prompt: str = "",
    action_feedback: str = "",
) -> Group:
    """Build the full terminal frame."""
    elements = []

    # ── header ──
    turn = int(state.current_player)
    actions = int(state.actions_taken)
    header = f"🚢 CONTAINER  │  Player {turn + 1}'s turn  │  Action {actions + 1}/2"
    if hist_msg:
        header += f"  │  {hist_msg}"
    elements.append(Panel(Text(header, style="bold white on blue")))

    # ── supply ──
    elements.append(Panel(_supply_bar(state, nc), title="Supply", border_style="yellow"))

    # ── player cards row ──
    cards = []
    for p in range(num_players):
        is_current = int(state.current_player) == p
        cards.append(_player_card(state, p, nc, is_current))

    from rich.columns import Columns
    elements.append(Columns(cards, equal=False, expand=True))

    # ── action feedback ──
    if action_feedback:
        elements.append(Panel(Text(action_feedback, style="green"), border_style="green"))

    # ── action help (with dynamic factory cost) ──
    fc = sum(1 for c in range(nc) if int(state.factory_colors[turn, c]) > 0)
    fac_cost = (fc + 1) * 3 if fc < MAX_FACTORIES_PER_PLAYER else 0
    elements.append(Panel(_action_help(fac_cost), title="Actions", border_style="cyan"))

    # ── prompt ──
    if prompt:
        elements.append(Panel(Text.from_markup(prompt, style="bold yellow"), border_style="yellow"))

    return Group(*elements)


# ── keyboard input ───────────────────────────────────────────────────────────


_stdin_fd: int = sys.stdin.fileno()
_orig_term_attrs: list | None = None  # saved terminal attrs for restore on exit


def _enter_raw_mode() -> None:
    """Enter cbreak mode once for the entire session. Must call _exit_raw_mode to restore."""
    global _orig_term_attrs
    _orig_term_attrs = termios.tcgetattr(_stdin_fd)
    tty.setcbreak(_stdin_fd, when=termios.TCSANOW)


def _exit_raw_mode() -> None:
    """Restore the terminal to its original mode."""
    global _orig_term_attrs
    if _orig_term_attrs is not None:
        termios.tcsetattr(_stdin_fd, termios.TCSANOW, _orig_term_attrs)
        _orig_term_attrs = None


def _read_key(timeout: float | None = None) -> str:
    """Read a single character, handling escape sequences.

    If timeout is None, blocks until a key is pressed.
    If timeout is a number, waits at most that many seconds, returning '' if none.
    """
    fd = _stdin_fd
    if timeout is None:
        r, _, _ = select.select([sys.stdin], [], [])
    else:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return ""
    ch = os.read(fd, 1).decode()
    if ch == "\x1b":
        r2, _, _ = select.select([sys.stdin], [], [], 0.05)
        if r2:
            ch2 = os.read(fd, 1).decode()
            if ch2 == "[":
                r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r3:
                    ch3 = os.read(fd, 1).decode()
                    return f"\x1b[{ch3}"
                return "\x1b["
            return ch + ch2
        return ch
    return ch


# ── input dialogs (work within the live display) ─────────────────────────────


def _input_number(prompt: str, max_val: int, live: Live, nc: int, np_: int, current_state) -> int | None:
    """Read a number 1-max_val via keystrokes while showing prompt in live display."""
    buf = ""
    while True:
        live.update(
            _render_frame(current_state, nc, np_, prompt=f"{prompt} [{buf}]")
        )
        live.refresh()
        ch = _read_key(10.0)
        if ch == "":
            continue
        if ch in ("\r", "\n"):
            if buf.isdigit():
                val = int(buf)
                if 1 <= val <= max_val:
                    return val
            # Invalid, clear buf
            buf = ""
        elif ch in ("\x7f", "\x08"):  # backspace
            buf = buf[:-1]
        elif ch in ("\x1b", "q", "Q"):  # ESC or q to cancel
            return None
        elif ch.isdigit():
            buf += ch
            # Auto-submit when a single digit is pressed and max is <= 9
            if len(buf) == 1 and max_val <= 9:
                val = int(buf)
                if 1 <= val <= max_val:
                    return val


def _input_choice(options: list[str], live: Live, nc: int, np_: int, current_state, header: str = "") -> int | None:
    """Present a list of options and let user select with number keys or ESC to cancel."""
    prompt_text = ""
    if header:
        prompt_text += header + "\n\n"
    prompt_text += "\n".join(f"  {i + 1}. {opt}" for i, opt in enumerate(options))
    prompt_text += "\n\n[dim]Number to select, ESC to cancel[/dim]"
    return _input_number(prompt_text, len(options), live, nc, np_, current_state)


# ── sub-action menu (choose colors, opponents, etc.) ─────────────────────────


def _submenu_buy_factory(state: EnvState, live: Live, nc: int, np_: int, num_players: int) -> dict | None:
    player = int(state.current_player)
    owned = {c for c in range(nc) if int(state.factory_colors[player, c]) > 0}
    fc = len(owned)
    cost = (fc + 1) * 3
    options = []
    option_colors = []
    for c in range(nc):
        emoji = COLOR_EMOJI[c] if c < len(COLOR_EMOJI) else ""
        status = " [dim](already owned)[/dim]" if c in owned else ""
        options.append(f"[{_cstyle(c)}]{emoji} {_cname(c, nc)}[/{_cstyle(c)}]{status}")
        option_colors.append(c)
    if not options:
        return None
    header = f"[bold]Buy a factory — cost: [green]${cost}[/green][/bold]"
    choice = _input_choice(options, live, nc, np_, state, header=header)
    if choice is None:
        return None
    return {"color": option_colors[choice - 1]}


def _submenu_pick_store(
    state: EnvState, target: int, store_type: str, live: Live, nc: int, np_: int
) -> dict | None:
    store = state.factory_store if store_type == "factory" else state.harbour_store
    available = []
    for c in range(nc):
        for s in range(PRICE_SLOTS):
            if int(store[target, c, s]) > 0:
                available.append((c, s))
    if not available:
        return None
    options = []
    for c, s in available:
        cnt = int(store[target, c, s])
        options.append(f"[{_cstyle(c)}]{cnt}× {_cname(c, nc)}[/{_cstyle(c)}] at [yellow]${s + 1}[/yellow]")
    choice = _input_choice(options, live, nc, np_, state)
    if choice is None:
        return None
    c, s = available[choice - 1]
    return {"color": c, "price_slot": s}


def _submenu_pick_opponent(state: EnvState, live: Live, nc: int, np_: int, num_players: int) -> int | None:
    player = int(state.current_player)
    candidates = _opponent_menu_indices(player, num_players)
    if len(candidates) == 1:
        return candidates[0]
    options = [f"Player {p + 1}" for p in candidates]
    choice = _input_choice(options, live, nc, np_, state)
    if choice is None:
        return None
    return candidates[choice - 1]


def _submenu_domestic_sale(state: EnvState, live: Live, nc: int, np_: int, num_players: int) -> dict | None:
    player = int(state.current_player)
    # First pick store type
    store_options = [
        "Factory Store (falls back to Harbour if empty)",
        "Harbour Store",
    ]
    choice = _input_choice(store_options, live, nc, np_, state)
    if choice is None:
        return None
    store_type = choice - 1

    own_store = state.factory_store if store_type == 0 else state.harbour_store
    available = []
    for c in range(nc):
        for s in range(PRICE_SLOTS):
            if int(own_store[player, c, s]) > 0:
                available.append((c, s))
    if not available and store_type == 0:
        own_store = state.harbour_store
        for c in range(nc):
            for s in range(PRICE_SLOTS):
                if int(own_store[player, c, s]) > 0:
                    available.append((c, s))
    if not available:
        return None
    options = [
        f"[{_cstyle(c)}]{int(own_store[player, c, s])}× {_cname(c, nc)}[/{_cstyle(c)}] at [yellow]${s + 1}[/yellow]"
        for c, s in available
    ]
    choice = _input_choice(options, live, nc, np_, state)
    if choice is None:
        return None
    c, s = available[choice - 1]
    return {"store_type": store_type, "color": c, "price_slot": s}


def _submenu_produce(
    state: EnvState, live: Live, nc: int, np_: int, num_players: int,
    env, encoder, history, rng_key,
) -> bool:
    """Collect per-color prices from the user and call env.step() for each factory colour.

    Returns True if the submenu should cancel (user pressed q/ESC).
    """
    player = int(state.current_player)
    factories = [c for c in range(nc) if int(state.factory_colors[player, c]) > 0]
    if not factories:
        return False

    import jax.numpy as jnp

    for color in factories:
        # Skip colours where supply is exhausted or factory space is full
        # (the user can still leave them idle via the menu option).
        if int(state.container_supply[color]) == 0:
            continue
        fc = int(jnp.sum(state.factory_colors[player]))
        capacity = fc * FACTORY_STORAGE_MULTIPLIER
        stored = int(jnp.sum(env.state.factory_store[player]))
        if stored >= capacity:
            continue

        header = f"[bold]Produce [{_cstyle(color)}]{_cname(color, nc)}[/{_cstyle(color)}] — pick a price:[/bold]"
        options = [f"${i + 1}" for i in range(PRODUCE_CHOICES - 1)] + ["[dim]leave idle[/dim]"]
        choice = _input_choice(options, live, nc, np_, env.state, header=header)
        if choice is None:
            return True
        if choice == len(options):
            price_slot = LEAVE_IDLE
        else:
            price_slot = choice - 1

        action_idx = encoder.encode(ACTION_PRODUCE, {"color": color, "price_slot": price_slot})
        obs, reward, term, trunc, info = env.step(action_idx)

        decoder = ActionEncoder(num_players, nc)
        try:
            decoded_type, decoded_params = decoder.decode(action_idx)
            desc = f"P{player + 1}: {_describe_action(decoded_type, decoded_params, nc)}"
        except Exception:
            desc = f"P{player + 1}: action #{action_idx}"

        history.append((env.state, desc, float(reward)))

    return False


def _submenu_buy_from_factory(
    state: EnvState, live: Live, nc: int, np_: int, num_players: int,
    env, encoder, history, rng_key,
) -> bool:
    """Pick an opponent and buy containers from their factory store per colour.

    Loops through each colour where the opponent has stock, lets the user pick
    a price (from what's available), and calls env.step() for each purchase.
    The first purchase consumes the action slot; subsequent purchases run as
    recurrent shopping steps.  Returns True if the submenu was cancelled
    before any purchase.
    """
    import jax.numpy as jnp

    player = int(state.current_player)

    target = _submenu_pick_opponent(state, live, nc, np_, num_players)
    if target is None:
        return True

    opp_idx = _opponent_menu_indices(player, num_players).index(target)

    factory_store = state.factory_store
    available_colors = [
        c for c in range(nc) if int(jnp.sum(factory_store[target, c])) > 0
    ]
    if not available_colors:
        return False

    first_purchase = True

    for color in available_colors:
        available_slots = [
            s for s in range(PRICE_SLOTS)
            if int(factory_store[target, color, s]) > 0
        ]
        if not available_slots:
            continue

        header = (
            f"[bold]Buy [{_cstyle(color)}]{_cname(color, nc)}[/{_cstyle(color)}] "
            f"from P{target + 1}'s factory — pick a price:[/bold]"
        )
        options = [f"${s + 1}" for s in available_slots]
        choice = _input_choice(options, live, nc, np_, env.state, header=header)
        if choice is None:
            if first_purchase:
                return True
            break
        price_slot = available_slots[choice - 1]

        action_idx = encoder.encode(
            ACTION_BUY_FROM_FACTORY_STORE,
            {"opponent": opp_idx + 1, "color": color, "price_slot": price_slot},
        )
        obs, reward, term, trunc, info = env.step(action_idx)
        first_purchase = False

        decoder = ActionEncoder(num_players, nc)
        try:
            decoded_type, decoded_params = decoder.decode(action_idx)
            desc = f"P{player + 1}: {_describe_action(decoded_type, decoded_params, nc)}"
        except Exception:
            desc = f"P{player + 1}: action #{action_idx}"

        history.append((env.state, desc, float(reward)))

        # Shopping may have auto-finished (no more space/cash/stock)
        if not int(env.state.shopping_active):
            return False

    # Send STOP if still shopping so the turn can advance
    if int(env.state.shopping_active) > 0:
        stop_action = jnp.array(
            [ACTION_BUY_FROM_FACTORY_STORE, 0, 0, 0, nc * PRICE_SLOTS],
            dtype=jnp.int32,
        )
        obs, reward, term, trunc, info = env.step(stop_action)
        history.append((env.state, f"P{player + 1}: Stop buying", float(reward)))

    return False


# ── describe actions ─────────────────────────────────────────────────────────


def _describe_action(action_type: int, params: dict, nc: int) -> str:
    if action_type == ACTION_BUY_FACTORY:
        return f"Buy {_cname(params.get('color', 0), nc)} factory"
    elif action_type == ACTION_BUY_WAREHOUSE:
        return "Buy warehouse"
    elif action_type == ACTION_PRODUCE:
        slot = params.get('price_slot', 0)
        if slot == LEAVE_IDLE:
            return f"Leave {_cname(params.get('color', 0), nc)} idle"
        return f"Produce {_cname(params.get('color', 0), nc)} at ${slot + 1}"
    elif action_type == ACTION_BUY_FROM_FACTORY_STORE:
        return f"Buy {_cname(params.get('color', 0), nc)} from P{params.get('opponent', 1)}'s factory at ${params.get('price_slot', 0) + 1}"
    elif action_type == ACTION_MOVE_LOAD:
        return f"Load {_cname(params.get('color', 0), nc)} from P{params.get('opponent', 1)}'s harbour"
    elif action_type == ACTION_MOVE_SEA:
        return "Move to Open Sea"
    elif action_type == ACTION_MOVE_AUCTION:
        return "Auction at Auction Island"
    elif action_type == ACTION_PASS:
        return "Pass"
    elif action_type == ACTION_TAKE_LOAN:
        return "Take loan"
    elif action_type == ACTION_REPAY_LOAN:
        return "Repay loan"
    elif action_type == ACTION_DOMESTIC_SALE:
        return f"Domestic sale of {_cname(params.get('color', 0), nc)}"
    return f"Action {action_type}"


# ── AI opponent ──────────────────────────────────────────────────────────────


def get_ai_action(state: EnvState, rng_key, num_players: int, num_colors: int):
    """Return a multi-head action array for the AI player."""
    import jax.numpy as jnp
    from jax import random

    player = int(state.current_player)
    encoder = ActionEncoder(num_players, num_colors)
    key, subkey = random.split(rng_key)

    produced = int(state.produced_this_turn) > 0
    produce_active = int(state.produce_active) > 0
    fc = int(jnp.sum(state.factory_colors[player]))
    wc = int(state.warehouse_count[player])
    cash = int(state.cash[player])
    loans = int(state.loans[player])
    has_space_f = int(jnp.sum(state.factory_store[player])) < fc * 2
    has_space_h = int(jnp.sum(state.harbour_store[player])) < wc

    def _encode(atype, params=None):
        return encoder.to_multi_head(encoder.encode(atype, params or {}))

    # If mid-produce batch, keep processing pending colours.
    if produce_active:
        for c in range(num_colors):
            if int(state.produce_pending[c]) > 0:
                return _encode(ACTION_PRODUCE, {"color": c, "price_slot": 1}), subkey
        # Fallback: nothing pending, pass to escape
        return _encode(ACTION_PASS), subkey

    if has_space_f and not produced:
        # AI produces the first owned factory color at $2
        owned_colors = [c for c in range(num_colors) if int(state.factory_colors[player, c]) > 0]
        ai_color = owned_colors[0] if owned_colors else 0
        return _encode(ACTION_PRODUCE, {"color": ai_color, "price_slot": 1}), subkey

    if loans > 0 and cash >= 11:
        return _encode(ACTION_REPAY_LOAN), subkey

    opp_indices = _opponent_menu_indices(player, num_players)
    if has_space_h:
        for opp in opp_indices:
            for c in range(num_colors):
                for s in range(PRICE_SLOTS):
                    if int(state.factory_store[opp, c, s]) > 0 and cash >= (s + 1):
                        params = {"opponent": opp_indices.index(opp) + 1, "color": c, "price_slot": s}
                        return _encode(ACTION_BUY_FROM_FACTORY_STORE, params), subkey

    owned = {c for c in range(num_colors) if int(state.factory_colors[player, c]) > 0}
    for c in range(num_colors):
        if c not in owned and fc < 5 and cash >= (fc + 1) * 2:
            return _encode(ACTION_BUY_FACTORY, {"color": c}), subkey

    if wc < 10 and cash >= (wc + 1):
        return _encode(ACTION_BUY_WAREHOUSE), subkey

    cargo = int(jnp.sum(state.ship_contents[player] > 0))
    if cargo > 0 and int(state.ship_location[player]) == LOCATION_OPEN_SEA:
        return _encode(ACTION_MOVE_AUCTION), subkey

    if cargo > 0 and int(state.ship_location[player]) >= LOCATION_HARBOUR_OFFSET:
        return _encode(ACTION_MOVE_SEA), subkey

    if cargo < SHIP_CAPACITY:
        for opp in opp_indices:
            for c in range(num_colors):
                for s in range(PRICE_SLOTS):
                    if int(state.harbour_store[opp, c, s]) > 0 and cash >= (s + 1):
                        params = {"opponent": opp_indices.index(opp) + 1, "color": c, "price_slot": s}
                        return _encode(ACTION_MOVE_LOAD, params), subkey

    if cash < 5 and loans < 2:
        return _encode(ACTION_TAKE_LOAN), subkey

    return _encode(ACTION_PASS), subkey


# ── main game command ────────────────────────────────────────────────────────


@app.command()
def play(
    num_players: int = typer.Option(2, "--players", "-p", help="Number of players"),
    num_colors: int = typer.Option(5, "--colors", "-c", help="Number of container colors"),
    human_players: str = typer.Option("0", "--humans", "-h", help="Comma-separated human player indices (0-based)"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
) -> None:
    """Play the Container board game in the terminal with a live TUI."""
    _enter_raw_mode()
    try:
        human_set = {int(x.strip()) for x in human_players.split(",") if x.strip()}
        for h in human_set:
            if h < 0 or h >= num_players:
                console.print(f"[red]Invalid human player index: {h}[/red]")
                raise typer.Exit(1)

        env = ContainerJaxEnv(num_players=num_players, num_colors=num_colors)
        encoder = ActionEncoder(num_players, num_colors)
        obs, info = env.reset(seed=seed)
        rng_key = jax.random.PRNGKey(seed)

        # state history: list of (state, description, reward)
        history: list[tuple[EnvState, str, float]] = [(env.state, "(start)", 0.0)]
        hist_idx = 0  # which history entry we're viewing (-1 = latest after game end)
        viewing_history = False
        done = False

        nc = num_colors
        np_ = num_players

        with Live(_render_frame(env.state, nc, np_), console=console, screen=True, auto_refresh=False) as live:
            while True:
                # If done (game over), always show final state
                display_state = env.state
                if viewing_history and hist_idx < len(history):
                    display_state = history[hist_idx][0]

                hist_msg = ""
                if len(history) > 1:
                    hist_msg = f"Step {hist_idx}/{len(history) - 1}  ← → to browse"
                    if viewing_history:
                        hist_msg += " [bold yellow](HISTORY VIEW - press → for latest)[/bold yellow]"

                live.update(
                    _render_frame(
                        display_state,
                        nc,
                        np_,
                        hist_msg=hist_msg,
                        action_feedback=history[hist_idx][1] if hist_idx < len(history) else "",
                    )
                )
                live.refresh()

                if done and not viewing_history:
                    # Show game over screen, wait for quit
                    ch = _read_key()
                    if ch in ("q", "\x1b"):
                        break
                    if ch == "\x1b[D" and len(history) > 1:  # left arrow
                        viewing_history = True
                        hist_idx = len(history) - 1
                    continue

                # Read keypress (poll every 50ms so the display stays live)
                ch = _read_key(timeout=0.05)
                if ch == "":
                    continue

                # ── global keys ──
                if ch in ("q", "Q"):
                    break

                # ── history navigation ──
                if ch == "\x1b[D":  # left arrow
                    if hist_idx > 0:
                        hist_idx -= 1
                        viewing_history = True
                    continue
                if ch == "\x1b[C":  # right arrow
                    if hist_idx < len(history) - 1:
                        hist_idx += 1
                        if hist_idx == len(history) - 1 and not done:
                            viewing_history = False
                    else:
                        viewing_history = False
                    continue

                # Can't act when viewing history
                if viewing_history:
                    continue

                if done:
                    continue

                state = env.state
                current = int(state.current_player)

                # ── action dispatch ──
                action_idx = None
                params = {}

                if ch in ("1", "2", "3", "4", "5", "6", "7", "p", "l", "r", "d", " "):
                    action_map = {
                        "1": ACTION_BUY_FACTORY,
                        "2": ACTION_BUY_WAREHOUSE,
                        "3": ACTION_PRODUCE,
                        "4": ACTION_BUY_FROM_FACTORY_STORE,
                        "5": ACTION_MOVE_LOAD,
                        "6": ACTION_MOVE_SEA,
                        "7": ACTION_MOVE_AUCTION,
                        "p": ACTION_PASS,
                        " ": ACTION_PASS,
                        "l": ACTION_TAKE_LOAN,
                        "r": ACTION_REPAY_LOAN,
                        "d": ACTION_DOMESTIC_SALE,
                    }
                    atype = action_map[ch]

                    # Sub-menus for actions that need parameters
                    if atype == ACTION_BUY_FACTORY:
                        result = _submenu_buy_factory(state, live, nc, np_, num_players)
                        if result is None:
                            continue
                        params = result
                    elif atype == ACTION_BUY_FROM_FACTORY_STORE:
                        cancelled = _submenu_buy_from_factory(
                            state, live, nc, np_, num_players,
                            env, encoder, history, rng_key,
                        )
                        if cancelled:
                            continue
                        hist_idx = len(history) - 1
                        rng_key, subkey = jax.random.split(rng_key)
                        if env.func_env.terminal(env.state, rng_key, env.func_env.params):
                            done = True
                        continue
                    elif atype == ACTION_MOVE_LOAD:
                        target = _submenu_pick_opponent(state, live, nc, np_, num_players)
                        if target is None:
                            continue
                        picked = _submenu_pick_store(state, target, "harbour", live, nc, np_)
                        if picked is None:
                            continue
                        opp_idx = _opponent_menu_indices(current, num_players).index(target)
                        params = {"opponent": opp_idx + 1, "color": picked["color"], "price_slot": picked["price_slot"]}
                    elif atype == ACTION_DOMESTIC_SALE:
                        result = _submenu_domestic_sale(state, live, nc, np_, num_players)
                        if result is None:
                            continue
                        params = result
                    elif atype == ACTION_PRODUCE:
                        cancelled = _submenu_produce(
                            state, live, nc, np_, num_players,
                            env, encoder, history, rng_key,
                        )
                        if cancelled:
                            continue
                        hist_idx = len(history) - 1
                        # check if game ended
                        rng_key, subkey = jax.random.split(rng_key)
                        if env.func_env.terminal(env.state, rng_key, env.func_env.params):
                            done = True
                        continue
                    elif atype in (ACTION_PASS, ACTION_TAKE_LOAN, ACTION_REPAY_LOAN,
                                   ACTION_MOVE_SEA, ACTION_MOVE_AUCTION, ACTION_BUY_WAREHOUSE):
                        pass

                    action_idx = encoder.encode(atype, params)
                else:
                    continue

                # ── either human or AI plays the action ──
                if current not in human_set:
                    # AI: override the human-selected action
                    action_idx, rng_key = get_ai_action(state, rng_key, num_players, num_colors)

                # Decode for description
                decoder = ActionEncoder(num_players, num_colors)
                try:
                    decoded_type, decoded_params = decoder.decode(action_idx)
                    desc = f"P{current + 1}: {_describe_action(decoded_type, decoded_params, nc)}"
                except Exception:
                    desc = f"P{current + 1}: action #{action_idx}"

                obs, reward, term, trunc, info = env.step(action_idx)
                reward_f = float(reward)

                # Check if game actually ended (env.state might be stale if term was already set)
                if env.func_env.terminal(env.state, rng_key, env.func_env.params):
                    term = True

                # Append to history
                history.append((env.state, desc, reward_f))
                hist_idx = len(history) - 1
                viewing_history = False

                if term or trunc:
                    done = True

        # Show final scores
        console.clear()
        state = env.state
        console.print("[bold green]═══ GAME OVER ═══[/bold green]\n")
        table = Table(title="Final Scores")
        table.add_column("Player")
        table.add_column("Cash")
        table.add_column("Net Worth", style="bold green")
        for p in range(num_players):
            cash = int(state.cash[p])
            nw = _compute_net_worth(state, p, nc)
            tag = " [cyan](you)[/cyan]" if p in human_set else ""
            table.add_row(f"Player {p + 1}{tag}", f"${cash}", f"[bold]${nw}[/bold]")
        console.print(table)
        winner = max(range(num_players), key=lambda p: _compute_net_worth(state, p, nc))
        console.print(f"\n[bold yellow]Player {winner + 1} wins! 🏆[/bold yellow]")
    finally:
        _exit_raw_mode()


@app.command()
def main() -> None:
    """Show help for Container RL CLI."""
    console.print("[bold]Container RL CLI[/bold]")
    console.print("Run [cyan]container-rl play[/cyan] to play the game.")


if __name__ == "__main__":
    app()
