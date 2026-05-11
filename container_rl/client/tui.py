"""Terminal UI for the Container game client."""

from __future__ import annotations

import fcntl
import os
import select
import sys
import termios
import time as _time
import tty
from typing import Any

import jax
import jax.numpy as jnp
from rich.console import Console, Group
from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from container_rl.client.client import GameClient
from container_rl.env.container import (
    ACTION_BUY_FACTORY,
    ACTION_BUY_FROM_FACTORY_STORE,
    ACTION_BUY_WAREHOUSE,
    ACTION_MOVE_AUCTION,
    ACTION_MOVE_LOAD,
    ACTION_MOVE_SEA,
    ACTION_PASS,
    ACTION_PRODUCE,
    ACTION_REPAY_LOAN,
    ACTION_TAKE_LOAN,
    HARBOUR_PRICE_CHOICES,
    HARBOUR_PRICE_MIN,
    LEAVE_IDLE,
    LOCATION_AUCTION_ISLAND,
    LOCATION_HARBOUR_OFFSET,
    LOCATION_OPEN_SEA,
    MAX_FACTORIES_PER_PLAYER,
    PRICE_SLOTS,
    PRODUCE_CHOICES,
    SHIP_CAPACITY,
    ActionEncoder,
    EnvState,
)
from container_rl.server.protocol import deserialize_state

jax.config.update("jax_disable_jit", True)
console = Console()

# ── connection state ───────────────────────────────────────────────────────
CLIENT: GameClient = None
GAME_ID: int | None = None
PLAYER_INDEX: int | None = None
GAME_CODE: str = ""
NUM_PLAYERS: int = 2
NUM_COLORS: int = 5
MY_NAME: str = ""
GAME_STATUS: str = "lobby"
PLAYER_NAMES: dict[int, str] = {}  # player_index -> name

# ── game state (received from server) ─────────────────────────────────────
STATE: EnvState | None = None
STATE_META: dict = {}
FEEDBACK: str = ""

# ── terminal helpers ─────────────────────────────────────────────────────

_ORIG_TERMIOS = None

def _enter_raw():
    global _ORIG_TERMIOS
    fd = sys.stdin.fileno()
    _ORIG_TERMIOS = termios.tcgetattr(fd)
    tty.setcbreak(fd)

def _exit_raw():
    if _ORIG_TERMIOS:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _ORIG_TERMIOS)

def _ch():
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        return sys.stdin.read(1)
    return ""

def _key(timeout: float | None = None) -> str:
    """Read a keystroke. timeout=None blocks indefinitely.  Returns escape
    sequences (e.g. ``'\\x1b[A'``) for arrow keys."""
    import os as _os
    if timeout is not None:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return ""
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        fd = sys.stdin.fileno()
        old_fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        try:
            fcntl.fcntl(fd, fcntl.F_SETFL, old_fl | os.O_NONBLOCK)
            extra = sys.stdin.read(5)
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, old_fl)
        if extra:
            return "\x1b" + extra
        return "\x1b"
    return ch

def _poll_server() -> dict | None:
    """Check for a server message without blocking.  Returns the parsed
    message, or None if nothing is available."""
    if CLIENT is None or CLIENT.sock is None:
        return None
    r, _, _ = select.select([CLIENT.sock], [], [], 0)
    if not r:
        return None
    try:
        return CLIENT.recv()
    except (ConnectionError, OSError):
        return {"type": "disconnected", "payload": {}}

def _drain_server() -> list[dict]:
    msgs = []
    while True:
        m = _poll_server()
        if m is None:
            break
        msgs.append(m)
    return msgs

def _send_action(action_idx: int) -> None:
    CLIENT.send("action", {"action_idx": action_idx})

def _send_multi_action(arr: list[int]) -> None:
    CLIENT.send("action_multi", {"action": arr})

# ── rendering ────────────────────────────────────────────────────────────

COLOR_NAMES = ["Red", "Green", "Blue", "Yellow", "Purple"]
COLOR_STYLES = ["red", "green", "blue", "yellow", "magenta"]

def _cn(idx, nc):
    return COLOR_NAMES[idx] if idx < len(COLOR_NAMES) else f"C{idx}"
def _cs(idx):
    return COLOR_STYLES[idx] if idx < len(COLOR_STYLES) else "white"

def _store_compact(store, player, nc):
    lines = []
    for c in range(nc):
        e = [f"{int(store[player,c,s])}×${s+1}" for s in range(PRICE_SLOTS) if int(store[player,c,s])>0]
        if e:
            lines.append(f"  [{_cs(c)}]{_cn(c,nc)}[/{_cs(c)}]: {', '.join(e)}")
    return "\n".join(lines) if lines else "[dim](empty)[/dim]"

def _island(store, player, nc):
    p = [f"[{_cs(c)}]{int(store[player,c])}× {_cn(c,nc)}[/{_cs(c)}]" for c in range(nc) if int(store[player,c])>0]
    return " ".join(p) if p else "[dim](empty)[/dim]"

def _ship(state, player):
    cargo = [int(state.ship_contents[player,i]) for i in range(SHIP_CAPACITY)]
    loc = int(state.ship_location[player])
    if loc == LOCATION_OPEN_SEA: ls = "[cyan]Open Sea[/cyan]"
    elif loc >= LOCATION_HARBOUR_OFFSET:
        hp = loc - LOCATION_HARBOUR_OFFSET
        ls = f"{PLAYER_NAMES.get(hp, f'P{hp+1}')}'s Harbour"
    elif loc == LOCATION_AUCTION_ISLAND: ls = "[yellow]Auction Isl.[/yellow]"
    else: ls = f"Loc {loc}"
    parts = ["·" if c==0 else f"[{_cs(c-1)}]■[/{_cs(c-1)}]" for c in cargo]
    return f"{' '.join(parts)}  @ {ls}"

def _net_worth(state, player, nc):
    cash = int(state.cash[player])
    hv = sum((s+1)*int(state.harbour_store[player,c,s]) for c in range(nc) for s in range(PRICE_SLOTS))
    sv = sum(3 for c in state.ship_contents[player] if int(c)>0)
    iv = 0
    for c in range(nc):
        cnt = int(state.island_store[player,c])
        if cnt>0:
            sec = int(state.secret_value_color[player])
            base = 10 if (c==sec and all(int(state.island_store[player,cc])>0 for cc in range(nc))) else (5 if c==sec else 2)
            iv += base*cnt
    return cash + hv + sv + iv - int(state.loans[player])*11

def _player_card(state, player, nc, is_current):
    cash = int(state.cash[player])
    loans = int(state.loans[player])
    wh = int(state.warehouse_count[player])
    sec = int(state.secret_value_color[player])
    nw = _net_worth(state, player, nc)
    facs = [f"[{_cs(c)}]{_cn(c,nc)}[/{_cs(c)}]" for c in range(nc) if int(state.factory_colors[player,c])]
    fstr = ", ".join(facs) if facs else "[dim]none[/dim]"
    badge = " [bold white on green]◄[/bold white on green]" if is_current else ""
    name = PLAYER_NAMES.get(player, f"P{player+1}")
    out = Text.from_markup(f"[bold]{name}{badge}[/bold]  ${nw}\n")
    out.append("─"*28+"\n")
    out.append_text(Text.from_markup(f"  💵 ${cash}  🏦 {loans} loans  🏭 {wh} wh\n"))
    out.append_text(Text.from_markup(f"  🤫 [{_cs(sec)}]{_cn(sec,nc)}[/{_cs(sec)}]\n"))
    out.append_text(Text.from_markup(f"  Factories: {fstr}\n"))
    out.append_text(Text.from_markup("  [bold]Factory Store:[/bold]\n"))
    out.append_text(Text.from_markup(_store_compact(state.factory_store,player,nc)+"\n"))
    out.append_text(Text.from_markup("  [bold]Harbour Store:[/bold]\n"))
    out.append_text(Text.from_markup(_store_compact(state.harbour_store,player,nc)+"\n"))
    out.append_text(Text.from_markup(f"  🏝️ {_island(state.island_store,player,nc)}\n"))
    out.append_text(Text.from_markup(f"  🚢 {_ship(state,player)}"))
    return out

def _supply(state, nc):
    parts = []
    ex = 0
    for c in range(nc):
        cnt = int(state.container_supply[c])
        if cnt<=0: parts.append(f"[{_cs(c)}]{_cn(c,nc)}[/{_cs(c)}]: [red]0[/red]"); ex+=1
        else: parts.append(f"[{_cs(c)}]{_cn(c,nc)}[/{_cs(c)}]: {'█'*min(cnt,10)} {cnt}")
    return Text.from_markup("  │  ".join(parts)+f"  │  [bold]Exhausted: {ex}/2[/bold]")

def _render(state, nc, np_, feedback="", my_player=None):
    elems = []
    turn = int(state.current_player)
    acts = int(state.actions_taken)
    name = PLAYER_NAMES.get(turn, f"Player {turn+1}")
    hdr = f"🚢 CONTAINER  │  {name}'s turn  │  Action {acts+1}/2"
    if int(state.auction_active): hdr += "  │  [yellow]AUCTION[/yellow]"
    if int(state.produce_active): hdr += "  │  [yellow]PRODUCING[/yellow]"
    if int(state.shopping_active): hdr += "  │  [yellow]SHOPPING[/yellow]"
    if my_player is not None and turn == my_player:
        hdr += "  [green](YOU)[/green]"
    elems.append(Panel(Text(hdr, style="bold white on blue")))
    elems.append(Panel(_supply(state,nc), title="Supply", border_style="yellow"))
    cards = [_player_card(state,p,nc,turn==p) for p in range(np_)]
    elems.append(Columns(cards, equal=False, expand=True))
    if feedback:
        elems.append(Panel(Text.from_markup(feedback, style="green"), border_style="green"))
    # action help
    if not int(state.auction_active) and not int(state.produce_active) and not int(state.shopping_active):
        fc = sum(1 for c in range(nc) if int(state.factory_colors[turn,c])>0)
        fac_c = (fc+1)*3 if fc<MAX_FACTORIES_PER_PLAYER else 0
        wh_c = int(state.warehouse_count[turn])+3
        fs = f"BuyFactory (${fac_c})" if fac_c else "BuyFactory"
        ws = f"BuyWarehouse (${wh_c})" if wh_c else "BuyWarehouse"
        ht = f" [1]{fs}  [2]{ws}  [3]Produce  [4]BuyFrFactry  [5]LoadShip\n [6]MoveToSea [7]Auction [0/space]Pass [8]TakeLoan [9]RepayLoan\n [q]uit"
        elems.append(Panel(Text.from_markup(ht), title="Actions", border_style="cyan"))
    elif int(state.auction_active):
        s = int(state.auction_seller); r = int(state.auction_round)
        t = f"Auction — P{s+1} selling" if r==0 else f"Auction — P{s+1}: accept/reject?"
        elems.append(Panel(Text(t, style="bold yellow"), title="Auction", border_style="yellow"))
    return Group(*elems)

# ── opponent helper ──────────────────────────────────────────────────────

def _opps(p, np_):
    return [(p+1+i)%np_ for i in range(np_-1)]

# ── input helpers ────────────────────────────────────────────────────────

def _input_number(live, state, nc, np_, prompt, max_val=999):
    buf = ""
    while True:
        live.update(_render(state, nc, np_, f"{prompt} [{buf}]", PLAYER_INDEX))
        live.refresh()
        ch = _key(10.0)
        if ch == "": continue
        if ch in ("\r","\n"):
            if buf.isdigit() and 1<=int(buf)<=max_val: return int(buf)
            buf=""
        elif ch in ("\x7f","\x08"): buf=buf[:-1]
        elif ch in ("\x1b","q","Q"): return None
        elif ch.isdigit():
            buf+=ch
            if len(buf)==1 and max_val<=9: return int(buf)
    return None

def _input_choice(live, state, nc, np_, options, header=""):
    p = header+"\n\n" if header else ""
    p += "\n".join(f"  {i+1}. {o}" for i,o in enumerate(options))
    p += "\n\n[dim]Number to select, ESC to cancel[/dim]"
    return _input_number(live, state, nc, np_, p, len(options))

def _pick_opponent(live, state, nc, np_):
    cand = _opps(int(state.current_player), np_)
    if len(cand)==1: return cand[0]
    ch = _input_choice(live, state, nc, np_, [PLAYER_NAMES.get(p, f"Player {p+1}") for p in cand])
    return cand[ch-1] if ch else None

# ── submenus ─────────────────────────────────────────────────────────────

def _desc_cargo(cargo, nc):
    parts = [f"[{_cs(int(cargo[i])-1)}]{_cn(int(cargo[i])-1,nc)}[/{_cs(int(cargo[i])-1)}]" for i in range(SHIP_CAPACITY) if int(cargo[i])>0]
    return " × ".join(parts) if parts else "(empty)"

def _submenu_produce(live, state, nc, np_):
    player = int(state.current_player)
    factories = [c for c in range(nc) if int(state.factory_colors[player,c])>0]
    if not factories: return True
    first = True
    for color in factories:
        hdr = f"[bold]Produce [{_cs(color)}]{_cn(color,nc)}[/{_cs(color)}] — pick a price:[/bold]"
        opts = [f"${i+1}" for i in range(PRODUCE_CHOICES-1)]+["[dim]leave idle[/dim]"]
        ch = _input_choice(live, state, nc, np_, opts, hdr)
        if ch is None:
            if first: return True
            break
        slot = LEAVE_IDLE if ch==len(opts) else ch-1
        arr = [ACTION_PRODUCE, 0, color, slot, 0]
        _send_multi_action(arr)
        first = False
        _wait_for_state(live, nc, np_)
        if STATE is None: return True
    # flush pending if cancelled mid-batch
    if STATE and int(STATE.produce_active)>0:
        for c in range(NUM_COLORS):
            if int(STATE.produce_pending[c])>0:
                _send_multi_action([ACTION_PRODUCE, 0, c, LEAVE_IDLE, 0])
                _wait_for_state(live, nc, np_)
    return False

def _submenu_buy_from_factory(live, state, nc, np_):
    global FEEDBACK
    player = int(state.current_player)
    target = _pick_opponent(live, state, nc, np_)
    if target is None: return True
    opp_idx = _opps(player, np_).index(target)
    first = True
    while True:
        if STATE is None: return True
        cheapest = {}
        for c in range(nc):
            for s in range(PRICE_SLOTS):
                if int(STATE.factory_store[target,c,s])>0:
                    if c not in cheapest or s<cheapest[c]: cheapest[c]=s
        if not cheapest: break
        clist = sorted(cheapest)
        opts = [f"[{_cs(c)}]{_cn(c,nc)}[/{_cs(c)}] at [yellow]${cheapest[c]+1}[/yellow]" for c in clist]+["[bold]Done[/bold]"]
        ch = _input_choice(live, STATE, nc, np_, opts, f"[bold]Buy from {PLAYER_NAMES.get(target, f'P{target+1}')}'s factory:[/bold]")
        if ch is None:
            if first: return True
            break
        if ch==len(opts): break
        color = clist[ch-1]; src_slot = cheapest[color]
        # harbour price
        hopts = [f"${s+1}" for s in range(HARBOUR_PRICE_MIN, HARBOUR_PRICE_MIN+HARBOUR_PRICE_CHOICES)]
        hch = _input_choice(live, STATE, nc, np_, hopts, f"[bold]Set harbour price for [{_cs(color)}]{_cn(color,nc)}[/{_cs(color)}]:[/bold]")
        if hch is None:
            if first: return True
            continue
        h_slot = HARBOUR_PRICE_MIN+hch-1
        arr = [ACTION_BUY_FROM_FACTORY_STORE, opp_idx, 0, h_slot, color*PRICE_SLOTS+src_slot]
        _send_multi_action(arr); first=False
        _wait_for_state(live, nc, np_)
        seller = PLAYER_NAMES.get(target, f"Player {target+1}")
        FEEDBACK = f"[bold]Buy {_cn(color,nc)} from {seller}'s factory at ${src_slot+1} (harbour ${h_slot+1})[/bold]"
        if STATE and not int(STATE.shopping_active):
            hs = sum(int(STATE.harbour_store[PLAYER_INDEX,c,s]) for c in range(nc) for s in range(PRICE_SLOTS))
            wh = int(STATE.warehouse_count[PLAYER_INDEX])
            if hs >= wh:
                msg = "[yellow]Harbour store full — buying stopped[/yellow]"
            else:
                msg = "[dim]No more containers available[/dim]"
            live.update(_render(STATE, nc, np_, msg, PLAYER_INDEX))
            live.refresh()
            _key(1.0)
            return False
    # STOP
    if STATE and int(STATE.shopping_active)>0:
        _send_multi_action([ACTION_BUY_FROM_FACTORY_STORE, 0, 0, 0, nc*PRICE_SLOTS])
        _wait_for_state(live, nc, np_)
    return False

def _submenu_move_load(live, state, nc, np_):
    global FEEDBACK
    player = int(state.current_player)
    target = _pick_opponent(live, state, nc, np_)
    if target is None: return True
    opp_idx = _opps(player, np_).index(target)
    first = True
    while True:
        if STATE is None: return True
        cheapest = {}
        for c in range(nc):
            for s in range(PRICE_SLOTS):
                if int(STATE.harbour_store[target,c,s])>0:
                    if c not in cheapest or s<cheapest[c]: cheapest[c]=s
        if not cheapest: break
        clist = sorted(cheapest)
        opts = [f"[{_cs(c)}]{_cn(c,nc)}[/{_cs(c)}] at [yellow]${cheapest[c]+1}[/yellow]" for c in clist]+["[bold]Done[/bold]"]
        ch = _input_choice(live, STATE, nc, np_, opts, f"[bold]Load from {PLAYER_NAMES.get(target, f'P{target+1}')}'s harbour:[/bold]")
        if ch is None:
            if first: return True
            break
        if ch==len(opts): break
        color = clist[ch-1]; slot = cheapest[color]
        _send_multi_action([ACTION_MOVE_LOAD, opp_idx, 0, 0, color*PRICE_SLOTS+slot])
        first=False
        _wait_for_state(live, nc, np_)
        seller = PLAYER_NAMES.get(target, f"Player {target+1}")
        FEEDBACK = f"[bold]Load {_cn(color,nc)} from {seller}'s harbour at ${slot+1}[/bold]"
        if STATE and not int(STATE.shopping_active):
            cargo = sum(1 for i in range(SHIP_CAPACITY) if int(STATE.ship_contents[PLAYER_INDEX,i])>0)
            if cargo >= SHIP_CAPACITY:
                msg = "[yellow]Ship cargo full — loading stopped[/yellow]"
            else:
                msg = "[dim]No more containers available[/dim]"
            live.update(_render(STATE, nc, np_, msg, PLAYER_INDEX))
            live.refresh()
            _key(1.0)
            return False
    # STOP
    if STATE and int(STATE.shopping_active)>0:
        _send_multi_action([ACTION_MOVE_LOAD, 0, 0, 0, nc*PRICE_SLOTS])
        _wait_for_state(live, nc, np_)
    return False

def _submenu_buy_factory(live, state, nc, np_):
    player = int(state.current_player)
    owned = {c for c in range(nc) if int(state.factory_colors[player,c])>0}
    fc = len(owned); cost = (fc+1)*3
    opts = []; colors = []
    for c in range(nc):
        s = " [dim](owned)[/dim]" if c in owned else ""
        opts.append(f"[{_cs(c)}]{_cn(c,nc)}[/{_cs(c)}]{s}")
        colors.append(c)
    ch = _input_choice(live, state, nc, np_, opts, f"[bold]Buy a factory — cost: [green]${cost}[/green][/bold]")
    if ch is None: return True
    return {"color": colors[ch-1]}

# ── state polling ────────────────────────────────────────────────────────

def _wait_for_state(live, nc, np_, timeout=5.0):
    global STATE, STATE_META, FEEDBACK
    deadline = _time.time()+timeout
    while _time.time()<deadline:
        for m in _drain_server():
            t = m.get("type", ""); p = m.get("payload", {})
            if t == "state_update":
                STATE_META = p
                if p.get("state"):
                    STATE = deserialize_state(bytes.fromhex(p["state"]))
                return STATE
            elif t == "action_result":
                FEEDBACK = f"[bold]{p.get('desc','')}[/bold]"
                if p.get("game_over"):
                    FEEDBACK += "  [bold red]GAME OVER[/bold red]"
            elif t == "error":
                FEEDBACK = f"[red]{p.get('message','')}[/red]"
            elif t == "disconnected":
                STATE = None
                return STATE
        live.update(_render(STATE, nc, np_, "Waiting for server…", PLAYER_INDEX))
        live.refresh()
        _time.sleep(0.05)
    return STATE

def _update_state_from_server(live, nc, np_):
    global STATE, STATE_META, FEEDBACK
    for m in _drain_server():
        t = m.get("type",""); p = m.get("payload",{})
        if t=="state_update":
            STATE_META=p
            if p.get("state"):
                STATE = deserialize_state(bytes.fromhex(p["state"]))
        elif t=="action_result":
            if p.get("game_over"): FEEDBACK="[bold red]GAME OVER[/bold red]"
            else: FEEDBACK=f"[bold]{p.get('desc','')}[/bold]"
        elif t=="error": FEEDBACK=f"[red]{p.get('message','')}[/red]"
        elif t=="disconnected": return None
    return STATE


# ── gameplay loop ────────────────────────────────────────────────────────

def _gameplay():
    global STATE, STATE_META, FEEDBACK
    # get initial state
    CLIENT.send("get_state")
    deadline = _time.time()+10
    while _time.time()<deadline:
        msgs = _drain_server()
        for m in msgs:
            if m.get("type")=="state_update":
                STATE = deserialize_state(bytes.fromhex(m["payload"]["state"]))
                STATE_META = m.get("payload",{})
                break
        if STATE: break
        _time.sleep(0.1)
    if STATE is None:
        console.print("[red]Failed to receive game state.[/red]"); _key(2); return

    encoder = ActionEncoder(NUM_PLAYERS, NUM_COLORS)

    with Live(_render(STATE, NUM_COLORS, NUM_PLAYERS,"",PLAYER_INDEX), console=console, screen=True, auto_refresh=False) as live:
        while True:
            _update_state_from_server(live, NUM_COLORS, NUM_PLAYERS)
            if STATE is None: return
            st = STATE
            cur = int(st.current_player)
            go = STATE_META.get("game_over",0)
            ac = int(st.auction_active)

            feedback_now = FEEDBACK
            FEEDBACK = ""

            if go:
                _key(3); return

            # ── auction mode ──
            if ac:
                seller = int(st.auction_seller); rnd = int(st.auction_round)
                if rnd == 0 and PLAYER_INDEX != seller:
                    # Any non-seller can bid — check if we already bid.
                    if int(st.auction_bids[PLAYER_INDEX]) < 0:
                        cargo = _desc_cargo(st.auction_cargo, NUM_COLORS)
                        bid = _input_number(live, st, NUM_COLORS, NUM_PLAYERS,
                            f"[bold]Auction![/bold] Cargo: {cargo}\n{PLAYER_NAMES.get(PLAYER_INDEX, f'P{PLAYER_INDEX+1}')}: bid (0=pass, max ${int(st.cash[PLAYER_INDEX])}):")
                        bid = bid if bid is not None else 0
                        _send_multi_action([ACTION_MOVE_AUCTION,PLAYER_INDEX,0,0,bid])
                        _wait_for_state(live, NUM_COLORS, NUM_PLAYERS)
                    else:
                        _update_state_from_server(live, NUM_COLORS, NUM_PLAYERS)
                        if STATE and int(STATE.auction_round) == 1:
                            pass  # will re-render on next iteration
                        else:
                            live.update(_render(STATE or st, NUM_COLORS, NUM_PLAYERS,
                                                "[dim]Bid submitted — waiting for others…[/dim]", PLAYER_INDEX))
                            live.refresh()
                            _time.sleep(0.1)
                elif rnd == 1 and PLAYER_INDEX == seller:
                    mx = max(0, int(jnp.max(st.auction_bids)))
                    ch = _input_choice(live,st,NUM_COLORS,NUM_PLAYERS,
                        [f"[green]Accept[/green] (+${mx}×2)",f"[red]Reject[/red] (-${mx})"],
                        f"[bold]Highest bid: ${mx}[/bold]")
                    acc = 1 if ch==1 else 0
                    _send_multi_action([ACTION_MOVE_AUCTION,PLAYER_INDEX,0,0,acc])
                    _wait_for_state(live, NUM_COLORS, NUM_PLAYERS)
                else:
                    _update_state_from_server(live, NUM_COLORS, NUM_PLAYERS)
                    if STATE:
                        live.update(_render(STATE, NUM_COLORS, NUM_PLAYERS,
                                            "[dim]Auction in progress…[/dim]", PLAYER_INDEX))
                        live.refresh()
                    _time.sleep(0.1)
                continue

            # ── produce/shopping modes — wait ──
            if int(st.produce_active) or int(st.shopping_active):
                if cur==PLAYER_INDEX:
                    if int(st.produce_active):
                        for c in range(NUM_COLORS):
                            if int(st.produce_pending[c])>0:
                                _send_multi_action([ACTION_PRODUCE,0,c,LEAVE_IDLE,0])
                                _wait_for_state(live,NUM_COLORS,NUM_PLAYERS); break
                    if int(st.shopping_active):
                        _send_multi_action([ACTION_BUY_FROM_FACTORY_STORE,0,0,0,NUM_COLORS*PRICE_SLOTS])
                        _wait_for_state(live,NUM_COLORS,NUM_PLAYERS)
                else:
                    _update_state_from_server(live, NUM_COLORS, NUM_PLAYERS)
                    if STATE:
                        live.update(_render(STATE, NUM_COLORS, NUM_PLAYERS,
                                            "[dim]Waiting for other player…[/dim]", PLAYER_INDEX))
                        live.refresh()
                    _time.sleep(0.1)
                continue

            # ── not our turn ──
            if cur!=PLAYER_INDEX and not ac:
                name = PLAYER_NAMES.get(cur, f"Player {cur+1}")
                while True:
                    _update_state_from_server(live, NUM_COLORS, NUM_PLAYERS)
                    if STATE is None: return
                    st = STATE; new_cur = int(st.current_player)
                    if new_cur == PLAYER_INDEX or int(st.auction_active):
                        break
                    name = PLAYER_NAMES.get(new_cur, f"Player {new_cur+1}")
                    live.update(_render(st, NUM_COLORS, NUM_PLAYERS,
                                        f"[dim]Waiting for {name} to play…[/dim]", PLAYER_INDEX))
                    live.refresh()
                    _time.sleep(0.1)
                continue

            live.update(_render(st, NUM_COLORS, NUM_PLAYERS, feedback_now, PLAYER_INDEX))
            live.refresh()

            # ── our turn: read action ──
            ch = _key(0.1)
            if ch=="": continue
            if ch in ("q","Q"): return
            if ch not in "0123456789 ": continue

            amap = {"0":ACTION_PASS," ":ACTION_PASS,"1":ACTION_BUY_FACTORY,"2":ACTION_BUY_WAREHOUSE,
                    "3":ACTION_PRODUCE,"4":ACTION_BUY_FROM_FACTORY_STORE,"5":ACTION_MOVE_LOAD,
                    "6":ACTION_MOVE_SEA,"7":ACTION_MOVE_AUCTION,"8":ACTION_TAKE_LOAN,"9":ACTION_REPAY_LOAN}
            atype = amap[ch]
            cancelled = False; aidx = None

            if atype == ACTION_BUY_FACTORY:
                r = _submenu_buy_factory(live, st, NUM_COLORS, NUM_PLAYERS)
                if r is True: continue
                if r: aidx = encoder.encode(atype, r)
            elif atype == ACTION_PRODUCE:
                cancelled = _submenu_produce(live, st, NUM_COLORS, NUM_PLAYERS)
            elif atype == ACTION_BUY_FROM_FACTORY_STORE:
                cancelled = _submenu_buy_from_factory(live, st, NUM_COLORS, NUM_PLAYERS)
            elif atype == ACTION_MOVE_LOAD:
                cancelled = _submenu_move_load(live, st, NUM_COLORS, NUM_PLAYERS)
            elif atype in (ACTION_BUY_WAREHOUSE,ACTION_MOVE_SEA,ACTION_MOVE_AUCTION,ACTION_PASS,ACTION_TAKE_LOAN,ACTION_REPAY_LOAN):
                aidx = encoder.encode(atype, {})

            if cancelled: continue
            if aidx is not None:
                _send_action(aidx)
                _wait_for_state(live, NUM_COLORS, NUM_PLAYERS)
                # Auction pre-check: warn if ship not at Open Sea or no cargo.
                if atype == ACTION_MOVE_AUCTION and STATE and not int(STATE.auction_active):
                    live.update(_render(STATE, NUM_COLORS, NUM_PLAYERS,
                                        "[yellow]Must be at Open Sea with cargo to hold an auction[/yellow]",
                                        PLAYER_INDEX))
                    live.refresh()
                    _key(1.5)


# ── main menu ────────────────────────────────────────────────────────────

def _read_line(prompt):
    console.print(f"[bold]{prompt}[/bold] ", end=""); sys.stdout.flush()
    buf=""
    while True:
        ch = _key(None)
        if ch=="\x1b": return None
        if ch in ("\r","\n"): console.print(); return buf.strip()
        if ch in ("\x7f","\x08"): buf=buf[:-1]; console.print(f"\r[bold]{prompt}[/bold] {buf} ",end=""); sys.stdout.flush()
        elif len(ch)==1 and ch.isprintable(): buf+=ch; console.print(ch,end=""); sys.stdout.flush()

def _read_password(prompt):
    console.print(f"[bold]{prompt}[/bold] ", end=""); sys.stdout.flush()
    buf=""
    while True:
        ch = _key(None)
        if ch=="\x1b": return None
        if ch in ("\r","\n"): console.print(); return buf.strip() or None
        if ch in ("\x7f","\x08"): buf=buf[:-1]
        elif len(ch)==1 and ch.isprintable(): buf+=ch

def _simple_choice(prompt, options):
    console.print(f"[bold]{prompt}[/bold]")
    for i,o in enumerate(options): console.print(f"  {i+1}. {o}")
    while True:
        ch = _key(None)
        if ch in ("\x1b","q","Q"): return None
        if ch.isdigit() and 1<=int(ch)<=len(options): return int(ch)

def _main_menu():
    """Highlighted main menu with ↑↓ selection."""
    opts = ["Create Game", "Join Game"]
    selected = 0
    while True:
        lines = []
        for i, o in enumerate(opts):
            prefix = "[bold yellow]>[/bold yellow]" if i == selected else " "
            style = "bold reverse" if i == selected else ""
            if style:
                lines.append(f"  {prefix} [{style}]{o}[/{style}]")
            else:
                lines.append(f"  {prefix} {o}")
        body = "\n".join(lines)
        frame = Text.from_markup(
            f"[bold]🚢 Container RL — select with ↑↓ or number, Enter to confirm:[/bold]\n\n{body}"
        )
        console.clear()
        console.print(Panel(frame, border_style="blue"))
        ch = _key(None)
        if ch == "\x1b":
            return None
        if ch in ("\x1b[A", "k", "w"):
            selected = max(0, selected - 1)
        elif ch in ("\x1b[B", "j", "s"):
            selected = min(len(opts) - 1, selected + 1)
        elif ch.isdigit():
            n = int(ch)
            if 1 <= n <= len(opts):
                return n
        elif ch in ("\r", "\n"):
            return selected + 1

def _create_screen():
    console.clear()
    console.print(Panel.fit("[bold]Create New Game[/bold]", border_style="green"))
    name = _read_line("Your name:")
    if name is None: return None
    pw = _read_password("Password (optional):")
    np_ = _simple_choice("Players:",["2","3","4","5","6"])
    if np_ is None: return None
    nc = _simple_choice("Colors:",["3","4","5"])
    if nc is None: return None
    return {"player_name":name,"password":pw,"num_players":np_+1,"num_colors":nc+2}

def _join_screen():
    console.clear()
    console.print(Panel.fit("[bold]Join Game[/bold]", border_style="green"))
    name = _read_line("Your name:")
    if name is None: return None
    pw = _read_password("Password (optional):")
    # fetch games
    CLIENT.send("list_games")
    _time.sleep(0.5)
    games = []
    for m in _drain_server():
        if m.get("type")=="game_list": games=m.get("payload",{}).get("games",[]); break

    code = _show_game_list(games)
    if code is None: return None
    return {"player_name":name,"password":pw,"code":code}


def _show_game_list(games: list[dict]) -> str | None:
    """Highlighted game list with ↑↓ selection.  Returns game code or None."""
    if not games:
        console.print("[dim]No open games found.[/dim]")
        return _read_line("Enter game code manually:")

    opts = [
        f"{g['code']}  [{g.get('slots_filled',0)}/{g.get('num_players','?')} joined]"
        f"{'  [yellow][in progress][/yellow]' if g.get('status') == 'active' else ''}"
        for g in games
    ]
    opts.append("Enter a game code manually …")

    selected = 0
    while True:
        lines = []
        for i, o in enumerate(opts):
            prefix = "[bold yellow]>[/bold yellow]" if i == selected else " "
            style = "bold reverse" if i == selected else ""
            if style:
                lines.append(f"  {prefix} [{style}]{o}[/{style}]")
            else:
                lines.append(f"  {prefix} {o}")
        body = "\n".join(lines)
        frame = Text.from_markup(
            f"[bold]Available games — ↑↓ to select, Enter to confirm, ESC to cancel:[/bold]\n\n{body}"
        )
        console.clear()
        console.print(Panel(frame, border_style="yellow"))
        ch = _key(None)
        if ch == "\x1b":
            return None
        if ch in ("\x1b[A", "k", "w"):
            selected = max(0, selected - 1)
        elif ch in ("\x1b[B", "j", "s"):
            selected = min(len(opts) - 1, selected + 1)
        elif ch.isdigit():
            n = int(ch)
            if 1 <= n <= len(games):
                return games[n - 1]["code"]
            elif n <= len(opts):
                selected = n - 1
        elif ch in ("\r", "\n"):
            break

    if selected < len(games):
        return games[selected]["code"]
    return _read_line("Enter game code:")

def _lobby():
    global PLAYER_NAMES, NUM_PLAYERS, GAME_CODE
    lobby_players = []
    while True:
        msgs = _drain_server()
        for m in msgs:
            t=m.get("type",""); p=m.get("payload",{})
            if t=="lobby_update":
                lobby_players = p.get("players",[])
                NUM_PLAYERS = p.get("num_players_needed", NUM_PLAYERS)
                PLAYER_NAMES = {int(pl["player_index"]): pl["name"] for pl in lobby_players}
            elif t=="game_started": return True
            elif t=="error": console.print(f"[red]{p.get('message','')}[/red]"); _key(1)
            elif t=="disconnected": return False
        console.clear()
        console.print(Panel.fit(f"[bold]Lobby — {GAME_CODE}[/bold]", border_style="yellow"))
        if lobby_players:
            console.print("[bold]Players:[/bold]")
            for pl in lobby_players:
                idx=pl.get("player_index",0); nm=pl.get("name","?")
                mrk=" [green](you)[/green]" if int(idx)==PLAYER_INDEX else ""
                console.print(f"  {nm}{mrk}")
        n = NUM_PLAYERS - len(lobby_players)
        console.print(f"\n[dim]{n} more needed.  q to leave.[/dim]")
        if _ch() in ("q","Q"): return False
        _time.sleep(0.3)
    return False


# ── entry point ──────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="Container game client")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=9876)
    args = p.parse_args()

    global CLIENT, GAME_ID, PLAYER_INDEX, GAME_CODE, NUM_PLAYERS, NUM_COLORS, MY_NAME, GAME_STATUS, PLAYER_NAMES
    CLIENT = GameClient(args.host, args.port)
    _enter_raw()
    try:
        try:
            CLIENT.connect()
        except Exception as e:
            console.print(f"[red]Cannot connect: {e}[/red]"); return

        ch = _main_menu()
        if ch is None: return

        if ch==1:
            cfg = _create_screen()
            if cfg is None: return
            MY_NAME=cfg["player_name"]
            CLIENT.send("create_game", cfg)
            for _i in range(50):
                msgs = _drain_server()
                for m in msgs:
                    if m.get("type")=="game_created":
                        pp=m["payload"]; GAME_ID=pp["game_id"]; PLAYER_INDEX=pp["player_index"]; GAME_CODE=pp["code"]
                        NUM_PLAYERS = cfg["num_players"]
                        NUM_COLORS = cfg["num_colors"]
                    if m.get("type")=="lobby_update":
                        pp=m["payload"]
                        NUM_PLAYERS = pp.get("num_players_needed", NUM_PLAYERS)
                        PLAYER_NAMES = {int(pl["player_index"]): pl["name"] for pl in pp.get("players", [])}
                if GAME_ID: break
                _time.sleep(0.1)
        else:
            cfg = _join_screen()
            if cfg is None: return
            MY_NAME=cfg["player_name"]
            CLIENT.send("join_game", cfg)
            for _i in range(50):
                msgs = _drain_server()
                for m in msgs:
                    if m.get("type")=="game_joined":
                        pp=m["payload"]; GAME_ID=pp["game_id"]; PLAYER_INDEX=pp["player_index"]; GAME_CODE=pp["code"]
                        NUM_PLAYERS=pp.get("num_players",NUM_PLAYERS); NUM_COLORS=pp.get("num_colors",NUM_COLORS)
                        GAME_STATUS=pp.get("status","lobby")
                    if m.get("type")=="lobby_update":
                        pp=m["payload"]
                        NUM_PLAYERS = pp.get("num_players_needed", NUM_PLAYERS)
                        PLAYER_NAMES = {int(pl["player_index"]): pl["name"] for pl in pp.get("players", [])}
                if GAME_ID: break
                _time.sleep(0.1)

        if GAME_ID is None:
            console.print("[red]Failed to create/join game.[/red]"); _key(2); return

        is_active = GAME_STATUS == "active"
        if not is_active:
            started = _lobby()
            if not started: return
        _gameplay()
    finally:
        _exit_raw()
        if CLIENT: CLIENT.disconnect()
        console.print("[dim]Goodbye![/dim]")

if __name__=="__main__": main()
