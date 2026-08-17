"""Entry point for the ``ledger`` command."""

from __future__ import annotations

import typer

from src import __version__
from src.cli.commands import (
    build,
    compare,
    fmt,
    gc,
    image,
    ingest,
    migrate,
    objects,
    read,
    vcs,
    verify,
)

app = typer.Typer(
    name="ledger",
    help="Ledger — a version control system for RL environments.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

app.add_typer(fmt.app)
app.add_typer(objects.app)
app.add_typer(ingest.app)
app.add_typer(read.app)
app.add_typer(vcs.app)
app.add_typer(compare.app)
app.add_typer(gc.app)
app.add_typer(image.app)
app.add_typer(build.app)
app.add_typer(verify.app)
app.add_typer(migrate.app)


@app.command()
def version() -> None:
    """Print the Ledger version."""
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
