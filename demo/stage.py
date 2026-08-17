"""Get this machine ready to record the demonstration, repeatably.

    uv run python demo/stage.py --check    # is this machine ready?
    uv run python demo/stage.py            # make it ready — run this between takes

The software working and the *recording* working are different problems, and the
second one has its own failure modes:

**Dead air.** Nothing worth watching should happen after the camera starts —
the corpus, the images and the port can all be settled first.

**State from the last take.** An environment that already exists, a port still
bound, a registry image still in the local Docker cache so ``docker pull`` prints
"up to date" instead of pulling on camera. Every retake has to begin identically,
or the run you keep is the one where you got lucky.

So this checks the preconditions, says precisely what is wrong when one is
missing, and resets everything a previous take touched.

Staging is **always** a clean slate — there is no separate "reset", because
wanting anything else between takes would mean recording against state you cannot
describe.

It deliberately does **not** build the Ledger corpus. The portal's first card
builds it live in a couple of seconds and every later card works on what that
card just made, which is the continuity worth having. Staging prepares only what
cannot be built on camera.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final

# Run as `python demo/stage.py` and the project root is not on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.e2e import build_environment

ROOT: Final = Path(__file__).resolve().parent.parent

#: Everything the demonstration generates, under one gitignored directory beside
#: the code that generates it. The leading dot is not decoration: ``demo/`` is
#: type-checked, the sample environment contains a generated ``verifier.py``, and
#: mypy skips dot-directories where it does not read ``.gitignore``.
DEMO_STATE: Final = Path(__file__).resolve().parent / ".state"

#: Where the demonstration keeps its corpus, and where the sample environment it
#: commits is built. Both are gitignored and both are removed by a reset.
DEMO_DATA: Final = DEMO_STATE / "data"
DEMO_WORKSPACE: Final = DEMO_STATE / "workspace"

#: The directory the portal's first card commits — the sample environment
#: itself, *inside* the workspace rather than the workspace.
#:
#: Named separately because the difference is one path segment and the symptom
#: is not: committing the workspace produces a tree with everything under
#: ``environment/``, so every later card reads ``data/train.bin`` and gets a 404.
#: It shipped that way once. There is one definition now, and
#: ``_build_workspace`` checks the builder agrees with it.
SAMPLE_ENVIRONMENT: Final = DEMO_WORKSPACE / "environment"

#: The port the recorded session uses. Fixed by default rather than found free,
#: because it gets typed into a browser on camera and a memorable one is worth
#: more than an automatic one. ``--port`` overrides it when this machine has 8080
#: already spoken for.
DEFAULT_PORT: Final = 8080

#: Base images the demonstration stores into Ledger. ``ledger image add --docker``
#: shells out to ``docker save``, which fails on an image this machine has never
#: pulled — so their absence is a precondition failure, not a runtime surprise.
BASE_IMAGES: Final = ("alpine:latest", "busybox:latest")


#: Images Ledger's own registry served during a previous take. Left in the local
#: cache, the first ``docker pull`` on camera prints "up to date" and the beat
#: shows nothing.
def served_prefix(port: int) -> str:
    return f"127.0.0.1:{port}/"


@final
@dataclass(frozen=True, slots=True)
class Check:
    """One precondition, and what to do when it is not met."""

    name: str
    ok: bool
    detail: str = ""
    #: The command that fixes it. Printed verbatim so it can be pasted.
    fix: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────────────────────


def preflight(port: int = DEFAULT_PORT) -> list[Check]:
    """Everything that has to be true before recording. Changes nothing."""
    return [
        _tool("uv", "the demonstration and the server are both run through it"),
        _tool("git", "the import beat converts a real repository"),
        _docker_daemon(),
        *[_base_image(name) for name in BASE_IMAGES],
        _port_free(port),
    ]


def _tool(name: str, why: str) -> Check:
    found = shutil.which(name)
    return Check(
        name=f"{name} on PATH",
        ok=found is not None,
        detail=found or why,
        fix="" if found else f"install {name}",
    )


def _docker_daemon() -> Check:
    """Binary *and* daemon.

    The same two-stage probe the end-to-end tests use, because "docker is
    installed" and "docker will answer" are different states and only the second
    one is any use — a stopped daemon otherwise fails halfway through the beat.
    """
    if shutil.which("docker") is None:
        return Check("docker daemon", ok=False, detail="no docker on PATH", fix="install Docker")
    probe = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return Check(
            "docker daemon",
            ok=False,
            detail="the CLI is installed but the daemon is not answering",
            fix="start Docker Desktop, then re-run",
        )
    return Check("docker daemon", ok=True, detail=f"server {probe.stdout.strip()}")


def _base_image(reference: str) -> Check:
    if shutil.which("docker") is None:
        return Check(f"{reference} cached", ok=False, detail="no docker", fix="install Docker")
    probe = subprocess.run(
        ["docker", "image", "inspect", reference],
        capture_output=True,
        check=False,
    )
    return Check(
        name=f"{reference} cached",
        ok=probe.returncode == 0,
        detail="" if probe.returncode == 0 else "not pulled on this machine",
        fix=f"docker pull {reference}",
    )


def _port_free(port: int) -> Check:
    """The port the browser will be pointed at, on camera.

    Names whatever holds it. "Something is already listening" sends you to
    ``lsof``; "node (pid 75381)" tells you immediately whether this is your own
    leftover ``ledgerd`` — kill it — or an unrelated service you would rather
    move around, and the two want opposite responses.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        taken = probe.connect_ex(("127.0.0.1", port)) == 0
    if not taken:
        return Check(f"port {port} free", ok=True)

    holder = _listening_on(port)
    ours = holder.startswith(("python", "uv", "ledgerd"))
    return Check(
        name=f"port {port} free",
        ok=False,
        detail=f"held by {holder}" if holder else "held by an unidentified process",
        fix=(
            "pkill -f 'ledgerd --data-dir'"
            if ours
            else f"stop it, or re-run with --port <free port> (lsof -nP -iTCP:{port} -sTCP:LISTEN)"
        ),
    )


def _listening_on(port: int) -> str:
    """``name (pid N)`` for whatever holds the port, or empty if it cannot say."""
    if shutil.which("lsof") is None:
        return ""
    probe = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-F", "cp"],
        capture_output=True,
        text=True,
        check=False,
    )
    pid = command = ""
    for line in probe.stdout.splitlines():
        if line.startswith("p"):
            pid = line[1:]
        elif line.startswith("c"):
            command = line[1:]
    return f"{command} (pid {pid})" if command else ""


# ─────────────────────────────────────────────────────────────────────────────
# Staging
# ─────────────────────────────────────────────────────────────────────────────


def stage(port: int, *, pull: bool = False) -> None:
    """Put the machine into the state every take starts from."""
    if pull:
        _pull_base_images()
    _clear_previous_take()
    _build_workspace()
    _forget_served_images(port)


def _pull_base_images() -> None:
    for reference in BASE_IMAGES:
        _say(f"pulling {reference}")
        subprocess.run(["docker", "pull", "--quiet", reference], check=False)


def _clear_previous_take() -> None:
    for directory in (DEMO_DATA, DEMO_WORKSPACE):
        if directory.exists():
            shutil.rmtree(directory)
            _say(f"removed {directory.relative_to(ROOT)}")


def _build_workspace() -> None:
    """The directory the hands-on beat commits.

    Built by the demonstration's own ``build_environment`` rather than by a second
    copy of it here: the two beats then show the *same* shape of environment, and
    there is one definition of what one looks like. It is seeded, so the byte
    range read on camera is identical in every take — which is how a retake is
    checkable rather than merely repeatable.
    """
    DEMO_WORKSPACE.mkdir(parents=True)
    source = build_environment(DEMO_WORKSPACE)
    if source != SAMPLE_ENVIRONMENT:  # pragma: no cover - guarded by a test
        raise AssertionError(
            f"the builder produced {source}, but the portal commits {SAMPLE_ENVIRONMENT}"
        )
    _say(f"built {source.relative_to(ROOT)}")


def _forget_served_images(port: int) -> None:
    """Drop images a previous take pulled *from Ledger*.

    Not the base images — those are expensive to fetch and are inputs. These are
    outputs, and leaving one cached makes the first pull on camera print "up to
    date", which shows nothing.
    """
    if shutil.which("docker") is None:
        return
    listed = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    served = [line for line in listed.stdout.split() if line.startswith(served_prefix(port))]
    if not served:
        return
    subprocess.run(["docker", "rmi", "-f", *served], capture_output=True, check=False)
    _say(f"forgot {len(served)} image(s) served by a previous take")


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def report(checks: list[Check]) -> bool:
    """Print the preflight table. Returns whether everything passed."""
    width = max(len(check.name) for check in checks)
    for check in checks:
        mark = "✓" if check.ok else "✗"
        line = f"  {mark}  {check.name:<{width}}"
        line = f"{line}   {check.detail}" if check.detail else line.rstrip()
        print(line)
        if not check.ok and check.fix:
            print(f"     {' ' * width}   → {check.fix}")
    return all(check.ok for check in checks)


def _say(message: str) -> None:
    print(f"  · {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Preflight only. Change nothing.")
    parser.add_argument("--pull", action="store_true", help="Fetch any missing base images.")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Server port. Default {DEFAULT_PORT}."
    )
    arguments = parser.parse_args()

    if not arguments.check:
        print("staging")
        stage(arguments.port, pull=arguments.pull)
        print()

    print("preflight")
    ready = report(preflight(arguments.port))
    print()

    if not ready:
        print("not ready to record — fix the above and re-run")
        return 1

    # Printed with the port *resolved*, so it is read from here rather than from
    # the runbook. A document cannot know which port this machine had free, and
    # a demonstration is not the moment to discover a stale number in one.
    print("ready.\n")
    print("    make demo                                  serve the portal and open it")
    print(f"    http://127.0.0.1:{arguments.port}/console                 …or open it yourself\n")
    print("On the page: 'Reset & run all' walks all twelve requirements.")
    print("Between takes: make demo\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
