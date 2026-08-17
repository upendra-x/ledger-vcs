"""What every command group needs: the console, the common options, and failure.

Error rendering is here rather than per-module because it is a *contract*, not a
convenience. A ``LedgerError`` carries a stable ``code`` and a ``details`` map
that the HTTP surface returns verbatim, and the point of printing both is that a
person debugging at the terminal and an automation reading a 409 body are looking
at the same thing. Four copies of that had already drifted apart in whitespace.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console

if TYPE_CHECKING:
    from src.errors import LedgerError

__all__ = ["DEFAULT_REF", "DataDir", "EnvOpt", "RefOpt", "console", "fail"]

console = Console()

DataDir = Annotated[Path, typer.Option("--data-dir", "-d", help="Ledger data directory.")]
EnvOpt = Annotated[str, typer.Option("--env", "-e", help="Environment name, org/name.")]
RefOpt = Annotated[str, typer.Option("--ref", "-r", help="Ref to operate on.")]

DEFAULT_REF = "refs/heads/main"


def fail(exc: LedgerError) -> typer.Exit:
    """Print an error the way the API returns it, and exit non-zero.

    Returns the exception to raise rather than raising, so call sites read
    ``raise fail(exc) from exc`` and keep the original traceback attached.
    """
    console.print(f"[red]{exc.code}[/red] {exc.message}")
    for key, value in exc.details.items():
        console.print(f"  [dim]{key}[/dim] {value}")
    return typer.Exit(1)
