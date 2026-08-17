"""``ledger image`` — put a container image inside a version.

The output is the layer *reuse* count and the compressed-versus-stored byte
counts, because those are the two numbers a claim rests on:
layers are stored uncompressed so that content-defined chunking works, and a
second image sharing a base layer costs nothing to add.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from src.cli._shared import DataDir, console
from src.errors import LedgerError
from src.ids import EnvName, RefName
from src.instance import Ledger
from src.oci.model import REF_NAME_ANNOTATION, Platform
from src.service.images import ImageService, host_platform
from src.text import human_bytes

app = typer.Typer()

image_app = typer.Typer(help="Store and inspect container images.")
app.add_typer(image_app, name="image")

DEFAULT_REF = "refs/heads/main"


@contextmanager
def _archive_for(source: str | None, docker: str | None) -> Iterator[Path]:
    """Yield a path to an image archive, exporting from Docker if asked.

    ``--docker`` is a convenience that shells out to ``docker save``. It is here
    rather than in the service because talking to a container daemon is a
    property of this machine, not of Ledger.
    """
    if (source is None) == (docker is None):
        raise typer.BadParameter("give either an archive path or --docker <image>")
    if source is not None:
        yield Path(source)
        return

    workspace = Path(tempfile.mkdtemp(prefix="ledger-image-"))
    try:
        archive = workspace / "image.tar"
        console.print(f"[dim]docker save {docker}[/dim]")
        result = subprocess.run(
            ["docker", "save", str(docker), "-o", str(archive)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise typer.BadParameter(f"docker save failed: {result.stderr.strip()}")
        yield archive
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@image_app.command("add")
def image_add(
    env: Annotated[str, typer.Argument(help="org/environment")],
    archive: Annotated[str | None, typer.Argument(help="An OCI layout or docker archive.")] = None,
    docker: Annotated[
        str | None, typer.Option("--docker", help="Export this image from the local daemon.")
    ] = None,
    name: Annotated[str, typer.Option("--name", "-n", help="Name inside the environment.")] = "app",
    ref: Annotated[str, typer.Option("--ref", "-r")] = DEFAULT_REF,
    data_dir: DataDir = Path("./data"),
    author: Annotated[str, typer.Option("--author")] = "ledger-cli",
    from_image: Annotated[
        str | None, typer.Option("--from", help="Which image in a multi-image archive.")
    ] = None,
    platform: Annotated[
        str | None, typer.Option("--platform", help="os/arch, e.g. linux/arm64.")
    ] = None,
) -> None:
    """Add an image to an environment as a new version.

    The commit *contains* the image, so restoring an old version restores the
    same layers — there is no tag that could have moved underneath it.
    """
    wanted = _platform(platform)
    try:
        with _archive_for(archive, docker) as path, Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            result = ImageService(ledger).add(
                env_id,
                RefName(ref),
                path,
                image=name,
                author=author,
                platform=wanted,
                from_image=from_image,
            )
    except LedgerError as exc:
        console.print(f"[red]{exc.code}[/red] {exc.message}")
        for key, value in exc.details.items():
            console.print(f"  [dim]{key}[/dim] {value}")
        raise typer.Exit(1) from exc

    console.print(f"[green]{result.commit}[/green]  [dim](commit)[/dim]\n")
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("image", f"{result.image} @ {result.manifest_digest}")
    table.add_row("layers", f"{result.layers} ({result.layers_reused} already stored)")
    table.add_row("compressed source", human_bytes(result.bytes_compressed))
    table.add_row("stored uncompressed", human_bytes(result.bytes_uncompressed))
    table.add_row(
        "objects created", f"{result.stats.objects_created} of {result.stats.objects_offered}"
    )
    table.add_row("bytes stored", human_bytes(result.stats.bytes_stored))
    console.print(table)
    console.print(
        f"\n[dim]docker pull <host>/{env}/{result.image}:{ref.removeprefix('refs/heads/')}[/dim]"
    )


@image_app.command("list")
def image_list(
    env: Annotated[str, typer.Argument(help="org/environment")],
    ref: Annotated[str, typer.Option("--ref", "-r")] = DEFAULT_REF,
    data_dir: DataDir = Path("./data"),
) -> None:
    """What images this version holds."""
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            index = ImageService(ledger).list_images(env_id, RefName(ref))
    except LedgerError as exc:
        console.print(f"[red]{exc.code}[/red] {exc.message}")
        raise typer.Exit(1) from exc

    if not index.manifests:
        console.print("[dim]this version holds no images[/dim]")
        return

    table = Table(box=None, pad_edge=False)
    table.add_column("image")
    table.add_column("digest", style="dim")
    table.add_column("platform")
    table.add_column("size", justify="right")
    for descriptor in index.manifests:
        table.add_row(
            descriptor.annotation_map.get(REF_NAME_ANNOTATION, "?"),
            str(descriptor.digest),
            str(descriptor.platform) if descriptor.platform else "-",
            human_bytes(descriptor.size),
        )
    console.print(table)


def _platform(text: str | None) -> Platform:
    if text is None:
        return host_platform()
    parts = text.split("/")
    if len(parts) < 2:
        raise typer.BadParameter("a platform is os/arch, e.g. linux/arm64")
    return Platform(
        os=parts[0], architecture=parts[1], variant=parts[2] if len(parts) > 2 else None
    )
