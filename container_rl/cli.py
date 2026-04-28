"""TUI for playing the Container board game via the terminal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
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
    LOCATION_AUCTION_ISLAND,
    LOCATION_HARBOUR_OFFSET,
    LOCATION_OPEN_SEA,
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

# ── colour palette for container colours ─────────────────────────────────────

COLOR_NAMES = ["Red", "Blue", "Green", "Yellow", "Purple"]
COLOR_STYLES = ["red", "blue", "green", "yellow", "magenta"]
COLOR_EMOJI = ["🔴", "🔵", "🟢", "🟡", "🟣"]


def _cname(idx: int, num_colors: int) -> str:
    if idx < len(COLOR_NAMES):
        return COLOR_NAMES[idx]
    return f"Color {idx}"


def _cstyle(idx: int) -> str:
    if idx < len(COLOR_STYLES):
        return COLOR_STYLES[idx]
    return "white"


# ── helpers ──────────────────────────────────────────────────────────────────


def _resolve_opponent(opp_idx: int, current_player: int, num_players: int) -> int:
    """Map opponent menu index (1-based, skipping current player) to player index."""
    candidates = [p for p in range(num_players) if p != current_player]
    if 1 <= opp_idx <= len(candidates):
        return candidates[opp_idx - 1]
    return (current_player + 1) % num_players


def _opponent_menu_indices(current_player: int, num_players: int) -> list[int]:
    return [p for p in range(num_players) if p != current_player]


# ── state rendering ──────────────────────────────────────────────────────────


def _render_store_table(store, player: int, num_colors: int) -> Table:
    """Render a store array as a Rich table: rows=colors, cols=price slots."""
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1), show_edge=False)
    table.add_column("Color", style="bold", width=8)
    for s in range(PRICE_SLOTS):
        table.add_column(f"${s + 1}", justify="right", width=4)
    table.add_column("Tot", justify="right", width=4)

    for c in range(num_colors):
        counts = [int(store[player, c, s]) for s in range(PRICE_SLOTS)]
        total = sum(counts)
        if total == 0:
            continue
        row = [f"[{_cstyle(c)}]{_cname(c, num_colors)}[/{_cstyle(c)}]"]
        for cnt in counts:
            row.append(str(cnt) if cnt > 0 else "·")
        row.append(str(total))
        table.add_row(*row)

    if len(table.rows) == 0:
        table.add_row("[dim](empty)[/dim]", *([""] * (PRICE_SLOTS + 1)))
    return table


def _render_store_compact(store, player: int, num_colors: int) -> str:
    """Render a store as a compact single-line-per-color format."""
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


def _render_island(island_store, player: int, num_colors: int) -> str:
    """Render island store as a compact string."""
    parts = []
    for c in range(num_colors):
        cnt = int(island_store[player, c])
        if cnt > 0:
            parts.append(f"[{_cstyle(c)}]{cnt}x {_cname(c, num_colors)}[/{_cstyle(c)}]")
    return "  ".join(parts) if parts else "[dim](empty)[/dim]"


def _render_ship(ship_contents, ship_location, player: int) -> str:
    """Render ship contents and location as a string."""
    row = ship_contents[player]
    loc = int(ship_location[player])
    parts = []
    for i in range(SHIP_CAPACITY):
        c = int(row[i])
        if c > 0:
            parts.append(f"[{_cstyle(c - 1)}]{'■'}[/{_cstyle(c - 1)}]")
        else:
            parts.append("·")
    cargo_str = " ".join(parts)

    if loc == LOCATION_OPEN_SEA:
        location_str = "[cyan]Open Sea[/cyan]"
    elif loc == LOCATION_AUCTION_ISLAND:
        location_str = "[yellow]Auction Island[/yellow]"
    elif loc >= LOCATION_HARBOUR_OFFSET:
        location_str = f"[green]Player {loc - LOCATION_HARBOUR_OFFSET + 1}'s Harbour[/green]"
    else:
        location_str = str(loc)

    return f"{cargo_str}  @  {location_str}"


def _render_player_panel(state: EnvState, player: int, num_colors: int, is_current: bool) -> Panel:
    """Render one player's full board."""
    cash = int(state.cash[player])
    loans = int(state.loans[player])
    warehouses = int(state.warehouse_count[player])
    secret = int(state.secret_value_color[player])

    factories = []
    for c in range(num_colors):
        if int(state.factory_colors[player, c]) > 0:
            factories.append(f"[{_cstyle(c)}]{_cname(c, num_colors)}[/{_cstyle(c)}]")
    factory_str = ", ".join(factories) if factories else "[dim]none[/dim]"

    nw = _compute_net_worth_simple(state, player, num_colors)

    header_parts = []
    if is_current:
        header_parts.append("[bold white on green] ► CURRENT TURN [/bold white on green] ")
    header_parts.append(f"[bold]Player {player + 1}[/bold]")
    title = " ".join(header_parts)

    elements: list = [
        Text.from_markup(f"💵 Cash: ${cash}  |  🏦 Loans: {loans}  |  🏭 Warehouses: {warehouses}"),
        Text.from_markup(f"🤫 Secret 10/5: [{_cstyle(secret)}]{_cname(secret, num_colors)}[/{_cstyle(secret)}]"),
        Text.from_markup(f"📊 Net Worth: ~${nw}"),
        Text.from_markup(f"🏭 Factories: {factory_str}"),
        Text(),
        Text("📦 Factory Store:", style="bold"),
        Text.from_markup(_render_store_compact(state.factory_store, player, num_colors)),
        Text(),
        Text("🏪 Harbour Store:", style="bold"),
        Text.from_markup(_render_store_compact(state.harbour_store, player, num_colors)),
        Text(),
        Text.from_markup(f"🏝️ Island: {_render_island(state.island_store, player, num_colors)}"),
        Text.from_markup(f"🚢 Ship: {_render_ship(state.ship_contents, state.ship_location, player)}"),
    ]

    style = "green" if is_current else "white"
    return Panel(Group(*elements), title=title, border_style=style)


def _render_supply(state: EnvState, num_colors: int) -> Panel:
    """Render container supply status."""
    parts = []
    exhausted = 0
    for c in range(num_colors):
        cnt = int(state.container_supply[c])
        if cnt <= 0:
            parts.append(f"[{_cstyle(c)}]{_cname(c, num_colors)}[/{_cstyle(c)}]: [red]EXHAUSTED[/red]")
            exhausted += 1
        else:
            parts.append(f"[{_cstyle(c)}]{_cname(c, num_colors)}[/{_cstyle(c)}]: {cnt}")
    text = "  ".join(parts)
    text += f"\n[bold]Exhausted colours: {exhausted}/2 needed to end game[/bold]"
    return Panel(text, title="Container Supply", border_style="yellow")


def _compute_net_worth_simple(state: EnvState, player: int, num_colors: int) -> int:
    """Quick net worth approximation for display (doesn't do full discard logic)."""
    cash = int(state.cash[player])
    loans_penalty = int(state.loans[player]) * 11
    harbour_val = int(jax.numpy.sum(state.harbour_store[player])) * 2
    ship_val = int(jax.numpy.sum(state.ship_contents[player] > 0)) * 3
    island_row = state.island_store[player]
    secret = int(state.secret_value_color[player])
    has_all = int(jax.numpy.all(island_row > 0))
    island_val = 0
    for c in range(num_colors):
        cnt = int(island_row[c])
        if cnt > 0:
            if c == secret:
                island_val += cnt * (10 if has_all else 5)
            else:
                island_val += cnt * 2
    return cash + harbour_val + ship_val + island_val - loans_penalty


def render_full_state(state: EnvState, num_players: int, num_colors: int, last_action: str = "") -> list:
    """Render the complete game state as a list of renderables."""
    out: list = []

    header_text = Text(
        f"🚢 CONTAINER  |  Turn: Player {int(state.current_player) + 1}  |  Actions taken: {int(state.actions_taken)}",
        style="bold white on blue",
    )
    if last_action:
        header_text.append(f"\n[dim]Last: {last_action}[/dim]")
    out.append(Panel(header_text))

    out.append(_render_supply(state, num_colors))

    for p in range(num_players):
        is_current = int(state.current_player) == p
        out.append(_render_player_panel(state, p, num_colors, is_current))

    out.append(Panel("Type /help for commands  |  /quit to exit", style="dim"))

    return out


# ── action selection ─────────────────────────────────────────────────────────


def _pick_color(num_colors: int, prompt: str = "Choose colour") -> int:
    console.print()
    for c in range(num_colors):
        emoji = COLOR_EMOJI[c] if c < len(COLOR_EMOJI) else ""
        console.print(f"  [{_cstyle(c)}]{c + 1}. {emoji} {_cname(c, num_colors)}[/{_cstyle(c)}]")
    choice = IntPrompt.ask(prompt, choices=[str(i + 1) for i in range(num_colors)], show_choices=False)
    return choice - 1


def _pick_price_slot(prompt: str = "Choose price slot") -> int:
    slot = IntPrompt.ask(f"{prompt} ($1-$10)", choices=[str(i) for i in range(1, PRICE_SLOTS + 1)], show_choices=False)
    return slot - 1


def _pick_opponent(current_player: int, num_players: int) -> int:
    candidates = _opponent_menu_indices(current_player, num_players)
    if len(candidates) == 1:
        return candidates[0]
    console.print()
    for i, p in enumerate(candidates):
        console.print(f"  {i + 1}. Player {p + 1}")
    choice = IntPrompt.ask("Choose opponent", choices=[str(i + 1) for i in range(len(candidates))], show_choices=False)
    return candidates[choice - 1]


def _show_store_preview(state: EnvState, player: int, num_colors: int, store_type: str = "factory") -> None:
    """Show a preview of another player's store to help with buying decisions."""
    store = state.factory_store if store_type == "factory" else state.harbour_store
    console.print(f"\n[bold]Player {player + 1}'s {store_type.title()} Store:[/bold]")
    table = _render_store_table(store, player, num_colors)
    console.print(table)


def _pick_store_container(state: EnvState, player: int, num_colors: int, store_type: str) -> dict | None:
    """Let user pick a container from another player's store. Returns params or None if cancelled."""
    store = state.factory_store if store_type == "factory" else state.harbour_store
    _show_store_preview(state, player, num_colors, store_type)

    available = []
    for c in range(num_colors):
        for s in range(PRICE_SLOTS):
            if int(store[player, c, s]) > 0:
                available.append((c, s))

    if not available:
        console.print("[red]No containers available in this store.[/red]")
        return None

    console.print("\n[bold]Available containers:[/bold]")
    for i, (c, s) in enumerate(available):
        cnt = int(store[player, c, s])
        console.print(
            f"  [{_cstyle(c)}]{i + 1}. {cnt}x {_cname(c, num_colors)}[/{_cstyle(c)}]"
            f" at [yellow]${s + 1}[/yellow]"
        )

    num_opts = len(available)
    choice = IntPrompt.ask(
        "Pick a container (or 0 to cancel)",
        choices=["0"] + [str(i + 1) for i in range(num_opts)],
        show_choices=False,
    )
    if choice == 0:
        return None
    c, s = available[choice - 1]
    return {"color": c, "price_slot": s}


def get_human_action(state: EnvState, encoder: ActionEncoder, num_players: int, num_colors: int) -> int:
    """Interactive action selection for a human player."""
    player = int(state.current_player)

    console.print(f"\n[bold]Player {player + 1}'s turn — choose an action:[/bold]\n")

    actions = [
        ("1", "Buy Factory", ACTION_BUY_FACTORY),
        ("2", "Buy Warehouse", ACTION_BUY_WAREHOUSE),
        ("3", "Produce containers", ACTION_PRODUCE),
        ("4", "Buy from Factory Store", ACTION_BUY_FROM_FACTORY_STORE),
        ("5", "Move to Harbour + Load", ACTION_MOVE_LOAD),
        ("6", "Move to Open Sea", ACTION_MOVE_SEA),
        ("7", "Move to Auction Island", ACTION_MOVE_AUCTION),
        ("p", "Pass [dim](Enter/space)[/dim]", ACTION_PASS),
        ("l", "Take Loan", ACTION_TAKE_LOAN),
        ("r", "Repay Loan", ACTION_REPAY_LOAN),
        ("d", "Domestic Sale", ACTION_DOMESTIC_SALE),
    ]

    for key, name, _ in actions:
        console.print(f"  [{key}] {name}")

    while True:
        raw = Prompt.ask("\nAction").strip().lower()
        if raw in ("", "p", " "):
            return encoder.encode(ACTION_PASS, {})
        for key, _name, atype in actions:
            if raw == key:
                action_type = atype
                break
        else:
            console.print("[red]Invalid choice. Try again (or Enter to Pass).[/red]")
            continue
        break

    params: dict = {}

    if action_type == ACTION_BUY_FACTORY:
        color = _pick_color(num_colors, "Choose factory colour")
        params = {"color": color}

    elif action_type == ACTION_BUY_WAREHOUSE:
        pass

    elif action_type == ACTION_PRODUCE:
        pass

    elif action_type == ACTION_BUY_FROM_FACTORY_STORE:
        target = _pick_opponent(player, num_players)
        picked = _pick_store_container(state, target, num_colors, "factory")
        if picked is None:
            return get_human_action(state, encoder, num_players, num_colors)
        opp_menu_idx = _opponent_menu_indices(player, num_players).index(target)
        params = {"opponent": opp_menu_idx + 1, "color": picked["color"], "price_slot": picked["price_slot"]}

    elif action_type == ACTION_MOVE_LOAD:
        target = _pick_opponent(player, num_players)
        picked = _pick_store_container(state, target, num_colors, "harbour")
        if picked is None:
            return get_human_action(state, encoder, num_players, num_colors)
        opp_menu_idx = _opponent_menu_indices(player, num_players).index(target)
        params = {"opponent": opp_menu_idx + 1, "color": picked["color"], "price_slot": picked["price_slot"]}

    elif action_type == ACTION_MOVE_SEA:
        pass

    elif action_type == ACTION_MOVE_AUCTION:
        pass

    elif action_type == ACTION_PASS:
        pass

    elif action_type == ACTION_TAKE_LOAN:
        pass

    elif action_type == ACTION_REPAY_LOAN:
        pass

    elif action_type == ACTION_DOMESTIC_SALE:
        console.print("\n[bold]Choose source:[/bold]")
        console.print("  1. Factory Store (falls back to Harbour if empty)")
        console.print("  2. Harbour Store")
        store_choice = IntPrompt.ask("Source", choices=["1", "2"], show_choices=False)
        store_type = store_choice - 1

        own_store = state.factory_store if store_type == 0 else state.harbour_store
        available = []
        for c in range(num_colors):
            for s in range(PRICE_SLOTS):
                cnt = int(own_store[player, c, s])
                if cnt > 0:
                    available.append((c, s))
        if not available and store_type == 0:
            own_store = state.harbour_store
            for c in range(num_colors):
                for s in range(PRICE_SLOTS):
                    if int(own_store[player, c, s]) > 0:
                        available.append((c, s))

        if not available:
            console.print("[red]No containers available for domestic sale.[/red]")
            return get_human_action(state, encoder, num_players, num_colors)

        console.print("\n[bold]Your containers:[/bold]")
        for i, (c, s) in enumerate(available):
            cnt = int(own_store[player, c, s])
            console.print(
                f"  [{_cstyle(c)}]{i + 1}. {cnt}x {_cname(c, num_colors)}[/{_cstyle(c)}]"
                f" at [yellow]${s + 1}[/yellow]"
            )
        num_opts = len(available)
        choice = IntPrompt.ask(
            "Pick a container (or 0 to cancel)",
            choices=["0"] + [str(i + 1) for i in range(num_opts)],
            show_choices=False,
        )
        if choice == 0:
            return get_human_action(state, encoder, num_players, num_colors)
        c, s = available[choice - 1]
        params = {"store_type": store_type, "color": c, "price_slot": s}

    return encoder.encode(action_type, params)


def _describe_action(action_type: int, params: dict, num_colors: int) -> str:
    """Return a human-readable description of an action."""
    if action_type == ACTION_BUY_FACTORY:
        return f"Buy {_cname(params.get('color', 0), num_colors)} factory"
    elif action_type == ACTION_BUY_WAREHOUSE:
        return "Buy warehouse"
    elif action_type == ACTION_PRODUCE:
        return "Produce containers"
    elif action_type == ACTION_BUY_FROM_FACTORY_STORE:
        opp = params.get("opponent", 1)
        c = params.get("color", 0)
        s = params.get("price_slot", 0)
        return f"Buy {_cname(c, num_colors)} from P{opp}'s factory @ ${s + 1}"
    elif action_type == ACTION_MOVE_LOAD:
        opp = params.get("opponent", 1)
        c = params.get("color", 0)
        s = params.get("price_slot", 0)
        return f"Load {_cname(c, num_colors)} from P{opp}'s harbour @ ${s + 1}"
    elif action_type == ACTION_MOVE_SEA:
        return "Move to Open Sea"
    elif action_type == ACTION_MOVE_AUCTION:
        return "Hold auction at Auction Island"
    elif action_type == ACTION_PASS:
        return "Pass"
    elif action_type == ACTION_TAKE_LOAN:
        return "Take loan"
    elif action_type == ACTION_REPAY_LOAN:
        return "Repay loan"
    elif action_type == ACTION_DOMESTIC_SALE:
        c = params.get("color", 0)
        return f"Domestic sale of {_cname(c, num_colors)}"
    return f"Action {action_type}"


# ── AI opponent ──────────────────────────────────────────────────────────────


def get_ai_action(state: EnvState, rng_key, num_players: int, num_colors: int) -> int:
    """Simple heuristic AI for non-human players."""
    import jax.numpy as jnp
    from jax import random

    player = int(state.current_player)
    encoder = ActionEncoder(num_players, num_colors)

    key, subkey = random.split(rng_key)

    produced = int(state.produced_this_turn) > 0
    factory_count = int(jnp.sum(state.factory_colors[player]))
    warehouse_count = int(state.warehouse_count[player])
    cash = int(state.cash[player])
    loans = int(state.loans[player])
    has_space_factory = int(jnp.sum(state.factory_store[player])) < factory_count * 2
    has_space_harbour = int(jnp.sum(state.harbour_store[player])) < warehouse_count

    if has_space_factory and not produced:
        return encoder.encode(ACTION_PRODUCE, {}), subkey

    if loans > 0 and cash >= 11:
        return encoder.encode(ACTION_REPAY_LOAN, {}), subkey

    if has_space_harbour:
        opp_indices = _opponent_menu_indices(player, num_players)
        for opp in opp_indices:
            for c in range(num_colors):
                for s in range(PRICE_SLOTS):
                    if int(state.factory_store[opp, c, s]) > 0 and cash >= (s + 1):
                        opp_menu_idx = opp_indices.index(opp)
                        params = {
                            "opponent": opp_menu_idx + 1,
                            "color": c,
                            "price_slot": s,
                        }
                        return encoder.encode(ACTION_BUY_FROM_FACTORY_STORE, params), subkey

    owned_colors = {c for c in range(num_colors) if int(state.factory_colors[player, c]) > 0}
    for c in range(num_colors):
        if c not in owned_colors and factory_count < 5 and cash >= (factory_count + 1) * 2:
            return encoder.encode(ACTION_BUY_FACTORY, {"color": c}), subkey

    if warehouse_count < 10 and cash >= (warehouse_count + 1):
        return encoder.encode(ACTION_BUY_WAREHOUSE, {}), subkey

    cargo_count = int(jnp.sum(state.ship_contents[player] > 0))
    if cargo_count > 0 and int(state.ship_location[player]) == LOCATION_OPEN_SEA:
        return encoder.encode(ACTION_MOVE_AUCTION, {}), subkey

    if cargo_count > 0 and int(state.ship_location[player]) >= LOCATION_HARBOUR_OFFSET:
        return encoder.encode(ACTION_MOVE_SEA, {}), subkey

    if cargo_count < SHIP_CAPACITY:
        opp_indices = _opponent_menu_indices(player, num_players)
        for opp in opp_indices:
            for c in range(num_colors):
                for s in range(PRICE_SLOTS):
                    if int(state.harbour_store[opp, c, s]) > 0 and cash >= (s + 1):
                        opp_menu_idx = opp_indices.index(opp)
                        params = {
                            "opponent": opp_menu_idx + 1,
                            "color": c,
                            "price_slot": s,
                        }
                        return encoder.encode(ACTION_MOVE_LOAD, params), subkey

    if cash < 5 and loans < 2:
        return encoder.encode(ACTION_TAKE_LOAN, {}), subkey

    return encoder.encode(ACTION_PASS, {}), subkey


# ── main game command ────────────────────────────────────────────────────────


@app.command()
def play(
    num_players: int = typer.Option(2, "--players", "-p", help="Number of players"),
    num_colors: int = typer.Option(5, "--colors", "-c", help="Number of container colors"),
    human_players: str = typer.Option("0", "--humans", "-h", help="Comma-separated human player indices (0-based)"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
) -> None:
    """Play the Container board game in the terminal."""
    human_set = {int(x.strip()) for x in human_players.split(",") if x.strip()}
    for h in human_set:
        if h < 0 or h >= num_players:
            console.print(f"[red]Invalid human player index: {h}. Must be 0-{num_players - 1}.[/red]")
            raise typer.Exit(1)

    if num_colors > 5:
        console.print(
            "[yellow]Warning: Only 5 colours have display names."
            " Extra colours will show as numbers.[/yellow]"
        )

    env = ContainerJaxEnv(num_players=num_players, num_colors=num_colors)
    encoder = ActionEncoder(num_players, num_colors)
    obs, info = env.reset(seed=seed)

    rng_key = jax.random.PRNGKey(seed)

    console.clear()
    console.print("[bold]Welcome to Container! 🚢[/bold]")
    console.print(f"Players: {num_players}  |  Colours: {num_colors}  |  Seed: {seed}")
    human_labels = [f"P{p + 1}" for p in sorted(human_set)]
    console.print(f"Human players: {', '.join(human_labels) if human_labels else 'none (AI-only)'}")
    console.print("\nPress Enter to start...")
    input()

    last_action_desc = ""
    step = 0
    done = False

    while not done:
        state = env.state
        current = int(state.current_player)
        is_terminal = bool(env.func_env.terminal(state, rng_key, env.func_env.params))

        if is_terminal:
            done = True
            break

        console.clear()
        for renderable in render_full_state(state, num_players, num_colors, last_action_desc):
            console.print(renderable)

        if current in human_set:
            action_idx = get_human_action(state, encoder, num_players, num_colors)
        else:
            action_idx, rng_key = get_ai_action(state, rng_key, num_players, num_colors)

        decoder = ActionEncoder(num_players, num_colors)
        try:
            atype, params = decoder.decode(action_idx)
            last_action_desc = f"P{current + 1}: {_describe_action(atype, params, num_colors)}"
        except Exception:
            last_action_desc = f"P{current + 1}: action #{action_idx}"

        obs, reward, term, trunc, info = env.step(action_idx)
        step += 1

        if term or trunc:
            done = True

    console.clear()
    state = env.state
    for renderable in render_full_state(state, num_players, num_colors, last_action_desc):
        console.print(renderable)

    console.print("\n[bold green]═══════════════════════════[/bold green]")
    console.print("[bold green]  GAME OVER!  [/bold green]")
    console.print("[bold green]═══════════════════════════[/bold green]\n")

    table = Table(title="Final Scores")
    table.add_column("Player", style="bold")
    table.add_column("Cash")
    table.add_column("Loans")
    table.add_column("Island")
    table.add_column("Harbour")
    table.add_column("Ship")
    table.add_column("Net Worth", style="bold green")

    for p in range(num_players):
        cash = int(state.cash[p])
        loans = int(state.loans[p])
        island_cnt = int(jax.numpy.sum(state.island_store[p]))
        harbour_cnt = int(jax.numpy.sum(state.harbour_store[p]))
        ship_cnt = int(jax.numpy.sum(state.ship_contents[p] > 0))
        nw = _compute_net_worth_simple(state, p, num_colors)
        table.add_row(
            f"Player {p + 1}" + (" [cyan](you)[/cyan]" if p in human_set else ""),
            f"${cash}",
            str(loans),
            str(island_cnt),
            str(harbour_cnt),
            str(ship_cnt),
            f"[bold]${nw}[/bold]",
        )

    console.print(table)

    winner = max(range(num_players), key=lambda p: _compute_net_worth_simple(state, p, num_colors))
    console.print(f"\n[bold yellow]Player {winner + 1} wins! 🏆[/bold yellow]")


@app.command()
def main() -> None:
    """Show help for Container RL CLI."""
    console.print("[bold]Container RL CLI[/bold]")
    console.print("Run [cyan]container-rl play[/cyan] to play the game.")


if __name__ == "__main__":
    app()
