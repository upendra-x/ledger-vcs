"""Entry point for ``ledgerd``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import uvicorn

app = typer.Typer(add_completion=False)


@app.command()
def serve(
    data_dir: Annotated[Path, typer.Option("--data-dir", "-d")] = Path("./data"),
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p")] = 8080,
    dev: Annotated[
        bool, typer.Option("--dev", help="Unauthenticated requests act as an administrator.")
    ] = False,
    workers: Annotated[int, typer.Option("--workers", "-w")] = 1,
) -> None:
    """Run the Ledger service."""
    from src.api.app import build_app

    uvicorn.run(
        build_app(data_dir, dev_mode=dev),
        host=host,
        port=port,
        workers=workers if workers > 1 else None,
        log_level="info",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
