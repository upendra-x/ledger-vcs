"""Start the demonstration: stage the machine, serve the portal, open it.

    make demo                       # or: uv run python demo/drive.py

There used to be a step-by-step driver here that typed CLI commands into a
terminal, and there is not one now. The demonstration lives in the browser, and
two demonstrations of one system are two things to keep true — the second one
silently rotting until the day it is shown.

So this is a launcher, and it does the three things that are easy to get wrong
in the two minutes before a recording starts:

* **stage**, so every take begins from the same clean slate;
* **preflight**, so a stopped Docker daemon or a held port is a message with a
  fix in it rather than a mystery on camera;
* **serve and open**, with the port resolved into the URL rather than left for
  somebody to remember.

Everything after that happens on the page. *Reset & run all* walks the whole
thing; otherwise open a card and run its steps one at a time.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Final

from rich.console import Console

# Run as `python demo/drive.py` and the project root is not on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.portal.app import Portal
from demo.stage import DEFAULT_PORT, SAMPLE_ENVIRONMENT, preflight, report, stage

__all__ = ["DEFAULT_PORT", "OPENING", "main"]

console = Console(highlight=False)

#: The opening, for the half minute while the page loads. The cards carry their
#: own explanations, so this is only the shape of the argument.
OPENING: Final = """\
Ledger is a version control system for RL environments, built for a corpus with
ten million of them — multi-gigabyte datasets, container images, and thousands
of automations writing at once.

One decision holds up the rest: state is split by how it changes. Immutable
content, named by the hash of its own bytes, and a small mutable layer of names
on top. A fork copies nothing, an old version brings back its own containers,
and two unrelated environments that share a dataset store it once — all of that
follows from the split rather than being built on top of it.

This page is organised by the twelve requirements. Every number on
it is measured during the run.\
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Server port. Default {DEFAULT_PORT}."
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="Keep whatever state is already there. Preflight still runs.",
    )
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser.")
    arguments = parser.parse_args()

    if not arguments.no_stage:
        console.print("\n[bold]staging[/bold]")
        stage(arguments.port)

    console.print("\n[bold]preflight[/bold]")
    if not report(preflight(arguments.port)):
        console.print("\n[red]not ready[/red] — fix the above and re-run\n")
        return 1

    url = f"http://127.0.0.1:{arguments.port}/console"
    console.print(f"\n[bold]the portal[/bold]   [green]{url}[/green]\n")
    console.print(f"[dim]{OPENING}[/dim]\n")
    console.print(
        "[dim]On the page: [/dim][bold]Reset & run all[/bold]"
        "[dim] walks all twelve, or open one card and run its steps by hand.[/dim]"
    )
    console.print("\n[dim]Ctrl-C to stop the server.[/dim]\n")

    if not arguments.no_open:
        _open_when_listening(url, arguments.port)

    import uvicorn

    portal = Portal(workspace=SAMPLE_ENVIRONMENT, port=arguments.port)
    try:
        uvicorn.run(portal.app, host="127.0.0.1", port=arguments.port, log_level="warning")
    except KeyboardInterrupt:  # pragma: no cover - the ordinary way to stop it
        pass
    finally:
        portal.close()
        console.print("\n[dim]server stopped[/dim]\n")
    return 0


def _open_when_listening(url: str, port: int, *, timeout_s: float = 15.0) -> None:
    """Open the browser once the socket answers, from a background thread.

    Opening it before the server binds shows an error page, and the fix —
    reload — is a small thing to do and a distracting one to do on camera.
    """

    def wait_and_open() -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                probe.settimeout(0.3)
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    webbrowser.open(url)
                    return
            time.sleep(0.2)

    threading.Thread(target=wait_and_open, daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
