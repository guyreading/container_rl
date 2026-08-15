"""Regression tests for the TUI living on the alternate screen buffer.

The whole TUI is drawn on the alternate screen so that leaving it restores the
shell untouched, with only "Goodbye!" printed after it.  Over ssh that is the
difference between logging out cleanly and leaving the board smeared across the
user's terminal.

``main`` enters that screen once and leaves it once.  The catch is ``Live``:
``Live(screen=True)`` treats the alternate screen as its own and leaves it on
exit whether or not it was the one that entered -- rich's ``set_alt_screen``
keeps no nesting count.  So when a game ended, we dropped back to the real
terminal and drew every menu after it (and the closing message) straight into
the user's scrollback.  ``_game_live`` re-enters behind ``Live`` to stop that.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from container_rl.client import tui

ENTER_ALT = "\x1b[?1049h"
LEAVE_ALT = "\x1b[?1049l"


@pytest.fixture
def screen(monkeypatch):
    """Swap in a console that records control codes instead of a real tty."""
    buf = io.StringIO()
    monkeypatch.setattr(tui, "console", Console(file=buf, force_terminal=True, width=80, height=24))
    return buf


def transitions(buf):
    """The alternate-screen switches in the order they were written."""
    text = buf.getvalue()
    return ["enter" if text[i + 7] == "h" else "leave" for i in _positions(text)]


def _positions(text):
    i = text.find("\x1b[?1049")
    while i >= 0:
        yield i
        i = text.find("\x1b[?1049", i + 8)


def on_alt_screen(buf):
    """Is the terminal on the alternate screen at the end of what we wrote?"""
    seen = transitions(buf)
    return bool(seen) and seen[-1] == "enter"


def test_alt_screen_round_trip(screen):
    tui._alt_screen(True)
    assert on_alt_screen(screen)
    tui._alt_screen(False)
    assert not on_alt_screen(screen)


def test_game_live_leaves_us_on_the_alt_screen(screen):
    """A finished game must not dump us back onto the real terminal."""
    tui._alt_screen(True)
    with tui._game_live("board"):
        pass
    assert on_alt_screen(screen), (
        "Live dropped the alternate screen on exit: every menu after a game "
        "would be drawn into the user's scrollback"
    )


def test_game_live_restores_alt_screen_when_gameplay_raises(screen):
    """Quitting mid-game unwinds through here too, so it gets the same care."""
    tui._alt_screen(True)
    with pytest.raises(RuntimeError):
        with tui._game_live("board"):
            raise RuntimeError("lost connection")
    assert on_alt_screen(screen)


def test_closing_message_lands_on_the_real_terminal(screen):
    """The full session shape: play a game, then quit."""
    tui._alt_screen(True)
    with tui._game_live("board"):
        pass
    tui.console.print("MENU")  # a menu drawn after the game -- must be hidden
    tui._alt_screen(False)
    tui.console.print("Goodbye!")

    text = screen.getvalue()
    last_leave = text.rfind(LEAVE_ALT)
    assert text.index("MENU") < last_leave, "menu was drawn on the real terminal"
    assert text.index("Goodbye!") > last_leave, "closing message was left on the alternate screen"
    assert not on_alt_screen(screen), "the TUI exited without restoring the terminal"
