"""The environment manifest — **the only module in Ledger that parses one**.

The *Format Agnostic* requirement is a claim about code that does not exist: the
storage layer must not know what the packaging format is. Ledger stores
``harbor.yaml`` exactly as it stores ``README.md`` — chunked, hashed, never
opened. Only this module opens it — the one place allowed to know what a
manifest is, and the reason nothing below it has to.

That is why replacing Harbor with something else changes no code below the API:
it changes this file, and this file is 150 lines.

The schema is deliberately small, and everything in it is optional. An
environment with no manifest is not an error — it is an environment that does
not build, which is most of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, final

import yaml

from src.errors import InvalidRequest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "MANIFEST_NAMES",
    "BuildStep",
    "Manifest",
    "describe",
    "find_manifest",
    "load_manifest",
    "parse_manifest",
]

#: Filenames recognised as an environment manifest, in precedence order. A list
#: rather than one name because a corpus of ten million environments migrated
#: from ten million repositories will not be uniform, and refusing to build the
#: ones that spell it differently is not a position worth defending.
MANIFEST_NAMES: Final = ("harbor.yaml", "harbor.yml", "environment.yaml", "ledger.yaml")

DEFAULT_TIMEOUT_SECONDS: Final = 900

#: A build may not take longer than this however the manifest is written. An
#: environment cannot be allowed to hold a worker forever: the pool is sized on
#: builds taking about five minutes, and an unbounded one turns a single bad
#: manifest into a capacity incident.
MAX_TIMEOUT_SECONDS: Final = 3600


@final
@dataclass(frozen=True, slots=True)
class BuildStep:
    """What to run, and for how long."""

    command: tuple[str, ...]
    workdir: str = "."
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    environment: tuple[tuple[str, str], ...] = ()

    @property
    def env_map(self) -> dict[str, str]:
        return dict(self.environment)


@final
@dataclass(frozen=True, slots=True)
class Manifest:
    """An environment's contribution to the shared pipeline."""

    name: str = ""
    schema_version: int = 1
    build: BuildStep | None = None
    #: Which images in ``images/index.json`` this environment publishes. Names,
    #: not digests: the digests are pinned by the commit already, and repeating
    #: them here would be a second place for them to disagree.
    images: tuple[str, ...] = ()
    #: Passed through to the platform verbatim. Ledger indexes nothing in here
    #: and validates nothing in here — modelling the platform's schema is
    #: exactly the coupling the sync contract exists to avoid.
    sync: tuple[tuple[str, str], ...] = ()

    @property
    def builds(self) -> bool:
        return self.build is not None

    @property
    def sync_metadata(self) -> dict[str, str]:
        return dict(self.sync)


def find_manifest(root: Path) -> Path | None:
    """The manifest in a materialized environment, or ``None`` if it has none."""
    for candidate in MANIFEST_NAMES:
        path = root / candidate
        if path.is_file():
            return path
    return None


def load_manifest(path: Path) -> Manifest:
    return parse_manifest(path.read_bytes())


def parse_manifest(data: bytes) -> Manifest:
    """Parse manifest bytes.

    ``yaml.safe_load`` rather than ``load``: a manifest is content a caller
    uploaded, and PyYAML's full loader constructs arbitrary Python objects. This
    is the one place in Ledger where uploaded bytes are interpreted at all, so it
    is the one place that could have that bug.
    """
    try:
        document = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        raise InvalidRequest("the environment manifest is not valid YAML", error=str(exc)) from exc

    if document is None:
        return Manifest()
    if not isinstance(document, dict):
        raise InvalidRequest("an environment manifest must be a mapping")

    return Manifest(
        name=str(document.get("name", "")),
        schema_version=int(document.get("version", 1)),
        build=_build_step(document.get("build")),
        images=_string_tuple(document.get("images"), "images"),
        sync=tuple(sorted(_string_map(document.get("sync"), "sync").items())),
    )


def _build_step(raw: Any) -> BuildStep | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InvalidRequest("'build' must be a mapping")

    command = raw.get("command")
    if command is None:
        raise InvalidRequest("a build step needs a 'command'")
    if isinstance(command, str):
        # A string is run through a shell, which is what people expect from
        # `command: make all`. A list is exec'd directly, with no shell to
        # reinterpret quoting — the safer form, and the one to prefer.
        argv: tuple[str, ...] = ("/bin/sh", "-c", command)
    elif isinstance(command, list) and command:
        argv = tuple(str(part) for part in command)
    else:
        raise InvalidRequest("'command' must be a non-empty string or list")

    timeout = int(raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    if timeout <= 0:
        raise InvalidRequest("'timeout_seconds' must be positive", timeout_seconds=timeout)

    return BuildStep(
        command=argv,
        workdir=str(raw.get("workdir", ".")),
        timeout_seconds=min(timeout, MAX_TIMEOUT_SECONDS),
        environment=tuple(sorted(_string_map(raw.get("env"), "build.env").items())),
    )


def _string_tuple(raw: Any, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if not isinstance(raw, list):
        raise InvalidRequest(f"'{field}' must be a string or a list of strings")
    return tuple(str(item) for item in raw)


def _string_map(raw: Any, field: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise InvalidRequest(f"'{field}' must be a mapping")
    return {str(key): str(value) for key, value in raw.items()}


def describe(manifest: Manifest) -> Mapping[str, Any]:
    """A summary safe to put in a build result. No command, no environment.

    A build result is world-readable within the corpus, and a manifest's
    environment map is exactly where someone will eventually put a token.
    """
    return {
        "name": manifest.name,
        "version": manifest.schema_version,
        "builds": manifest.builds,
        "images": list(manifest.images),
    }
