"""Maintainer TUI — manage games on the server."""

from __future__ import annotations

import argparse
import fcntl
import os
import select
import sys
import termios
import time as _time
import tty
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from container_rl.client.client import GameClient

console = Console()
CLIENT: GameClient = None

_ORIG_TERMIOS = None


def _enter_raw():
    global _ORIG_TERMIOS
    fd = sys.stdin.fileno()
    _ORIG_TERMIOS = termios.tcgetattr(fd)
    tty.setcbreak(fd)


def _exit_raw():
    if _ORIG_TERMIOS:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _ORIG_TERMIOS)


def _key(timeout: float | None = None) -> str:
    """Read a keystroke. timeout=None blocks indefinitely."""
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


def _drain_server(timeout: float = 0.5) -> list[dict]:
    """Read all available server messages, waiting up to *timeout* seconds."""
    if CLIENT is None or CLIENT.sock is None:
        return []
    import select as _sel
    msgs = []
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        r, _, _ = _sel.select([CLIENT.sock], [], [], max(0, deadline - _time.time()))
        if not r:
            break
        try:
            CLIENT.sock.setblocking(True)
            m = CLIENT.recv()
            if m is None:
                break
            msgs.append(m)
        except (ConnectionError, OSError):
            msgs.append({"type": "disconnected", "payload": {}})
            break
    return msgs


def _draw_list(games: list[dict], selected: int, feedback: str = "") -> None:
    lines: list[str] = []
    lines.append("")
    lines.append("[bold]Game Maintainer[/bold] — ↑↓ to navigate, [red]d[/red] to delete, [bold]r[/bold] to refresh, [bold]q[/bold] to quit")
    if feedback:
        lines.append(f"\n{feedback}")
    lines.append("")
    if not games:
        lines.append("[dim]No games found.[/dim]")
    else:
        for i, g in enumerate(games):
            code = g.get("code", "?")
            status = g.get("status", "?")
            css = {"lobby": "yellow", "active": "green", "finished": "dim"}.get(status, "white")
            slots = f"{g.get('slots_filled','?')}/{g.get('num_players','?')}"
            created = (g.get("created_at", "") or "")[:16]
            marker = "[bold yellow]>[/bold yellow]" if i == selected else " "
            row = f"  {marker} [{css}]{status:8}[/{css}]  {code:15}  {slots:7}  {created}"
            if i == selected:
                row = f"[bold reverse]{row}[/bold reverse]"
            lines.append(row)
    lines.append("")
    content = "\n".join(lines)
    console.clear()
    console.print(Panel(Text.from_markup(content), border_style="blue"))


def main():
    parser = argparse.ArgumentParser(description="Container game maintainer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--token", default="", help="Maintainer token from server startup")
    args = parser.parse_args()

    token = args.token.strip()
    if not token:
        token = input("Maintainer token: ").strip()
    if not token:
        console.print("[red]Token required.[/red]")
        return

    global CLIENT
    CLIENT = GameClient(args.host, args.port)
    _enter_raw()
    try:
        try:
            CLIENT.connect()
        except Exception as e:
            console.print(f"[red]Cannot connect: {e}[/red]")
            return

        games: list[dict] = []
        selected = 0
        feedback = ""

        def _load_games():
            nonlocal games, selected, feedback
            CLIENT.send("maintainer_list", {"token": token})
            for m in _drain_server(1.0):
                if m.get("type") == "maintainer_list":
                    games = m.get("payload", {}).get("games", [])
                    selected = min(selected, max(0, len(games) - 1))
                    return True
                elif m.get("type") == "error":
                    feedback = f"[red]{m['payload'].get('message', '')}[/red]"
                    return False
                elif m.get("type") == "disconnected":
                    feedback = "[red]Disconnected.[/red]"
                    return False
            return False

        _load_games()

        while True:
            _draw_list(games, selected, feedback)
            feedback = ""
            ch = _key(None)
            if ch in ("q", "Q", "\x1b"):
                break
            elif ch in ("\x1b[A", "k", "w"):
                selected = max(0, selected - 1)
            elif ch in ("\x1b[B", "j", "s"):
                selected = min(len(games) - 1, selected + 1)
            elif ch in ("d", "D"):
                if 0 <= selected < len(games):
                    g = games[selected]
                    _draw_list(games, selected,
                               f"[bold red]Delete {g['code']}?  \\[y]es / \\[n]o[/bold red]")
                    confirm = _key(None)
                    if confirm in ("y", "Y"):
                        CLIENT.send("maintainer_delete", {"token": token, "game_id": g["id"]})
                        for m in _drain_server(1.0):
                            if m.get("type") == "maintainer_result":
                                feedback = f"[green]Deleted game {m['payload'].get('deleted','?')}.[/green]"
                            elif m.get("type") == "error":
                                feedback = f"[red]{m['payload'].get('message','')}[/red]"
                        _load_games()
            elif ch in ("r", "R"):
                _load_games()
    finally:
        _exit_raw()
        CLIENT.disconnect()
        console.print("[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
