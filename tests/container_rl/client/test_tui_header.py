"""Regression tests for the game's top line.

The header is the one line every player looks at to know whose turn it is, and
it is built as a markup string: ``[green](YOU)[/green]`` beside your own name,
``[yellow]AUCTION[/yellow]`` while a sale is running.  Markup only becomes
colour if the string is parsed as markup -- handing it to a plain ``Text`` puts
the tags on screen verbatim, so the line the player reads is

    CONTAINER  |  GuyAR's turn  |  Action 1/2  [green](YOU)[/green]

which is how it looked when the header was drawn as
``Text(hdr, style="bold white on blue")``: literal tags, on a blue band that
fought with every other panel on the board.  ``Text.from_markup`` is what makes
the tags render, and these tests pin that down -- the header has been quietly
reverted to a plain ``Text`` once already, and over ssh the top line is the
first thing anyone sees.

The name in the middle is supplied by the player at registration, so it is
``escape``d before the header is parsed: a player called ``[red]x[/red]`` must
read as those characters, not repaint the line.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from rich.console import Console

from container_rl.client import tui
from container_rl.env.container import ContainerFunctional, ContainerParams

jax.config.update("jax_disable_jit", True)

NC = 5
NP = 3

# ── ANSI the header must and must not carry ──────────────────────────────
BLUE_BACKGROUND = "44m"   # the band that used to sit behind the top line
GREEN = "32m"
YELLOW = "33m"


@pytest.fixture
def state():
    params = ContainerParams(num_players=NP, num_colors=NC)
    env = ContainerFunctional()
    return env.initial(jax.random.PRNGKey(0), params)


@pytest.fixture(autouse=True)
def names(monkeypatch):
    monkeypatch.setattr(tui, "PLAYER_NAMES", {i: f"Player {i+1}" for i in range(NP)})


def header_line(state, *, my_player=None, color=True):
    """The rendered top line, with ANSI kept (``color``) or stripped."""
    console = Console(force_terminal=color, no_color=not color, width=100, height=40)
    with console.capture() as cap:
        console.print(tui._render(state, NC, NP, my_player=my_player))
    for line in cap.get().splitlines():
        if "CONTAINER" in line:
            return line
    raise AssertionError("no header line in the rendered frame")


def test_your_turn_marker_is_green_not_literal_markup(state):
    """``(YOU)`` renders as colour; the tags never reach the screen."""
    line = header_line(state, my_player=int(state.current_player))

    assert "(YOU)" in line
    assert "[green]" not in line
    assert "[/green]" not in line
    assert GREEN in line


def test_header_has_no_blue_band(state):
    """The top line is bold on the terminal's own background."""
    assert BLUE_BACKGROUND not in header_line(state, my_player=0)


def test_auction_tag_renders_as_colour(state):
    """The mode tags are markup too, and go the same way as ``(YOU)``."""
    line = header_line(state._replace(auction_active=jnp.array(1, dtype=jnp.int32)))

    assert "AUCTION" in line
    assert "[yellow]" not in line
    assert YELLOW in line


def test_player_name_is_shown_literally_not_as_markup(state, monkeypatch):
    """A name full of tags is text: it neither vanishes nor recolours the line."""
    turn = int(state.current_player)
    monkeypatch.setattr(
        tui, "PLAYER_NAMES", {turn: "[red]x[/red]", **{p: f"P{p+1}" for p in range(NP) if p != turn}}
    )

    plain = header_line(state, color=False)
    assert "[red]x[/red]'s turn" in plain
