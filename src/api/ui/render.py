"""HTML, assembled by hand.

There is no template engine here on purpose: the pages are tables over data the
services already produce, and a template language would be a second syntax to
learn for no expressive gain.

**Everything interpolated goes through ``e()``.** That is the entire safety story
and it has to be absolute, because a great deal of what these pages display is
attacker-controlled: environment names, commit messages, and file paths are all
content somebody uploaded. ``Markup`` exists for the few places that genuinely
need pre-built HTML, and it is only ever constructed from constant strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Final, final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ["Markup", "column", "e", "layout", "link", "short", "table"]


@final
@dataclass(frozen=True, slots=True)
class Markup:
    """HTML that has already been made safe. Never built from user input."""

    value: str

    def __str__(self) -> str:
        return self.value


def e(value: object) -> str:
    """Escape anything for HTML. The only way user data reaches a page."""
    if isinstance(value, Markup):
        return value.value
    return escape(str(value), quote=True)


STYLE: Final = """
:root { color-scheme: light dark; --line: #8883; --dim: #8888; --accent: #4f7cff; }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
main { max-width: 62rem; margin: 0 auto; padding: 1.5rem 1.25rem 4rem; }
header { border-bottom: 1px solid var(--line); padding: .85rem 1.25rem; }
header .inner { max-width: 62rem; margin: 0 auto; display: flex; gap: .75rem;
  align-items: baseline; }
header b { font-weight: 600; letter-spacing: .02em; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 1.15rem; margin: 0 0 .25rem; font-weight: 600; }
h2 { font-size: .95rem; margin: 2rem 0 .5rem; font-weight: 600; }
.dim { color: var(--dim); }
.hash { font-size: .82rem; color: var(--dim); word-break: break-all; }
table { border-collapse: collapse; width: 100%; margin: .25rem 0 1rem; }
th { text-align: left; font-weight: 600; font-size: .78rem; text-transform: uppercase;
  letter-spacing: .04em; color: var(--dim); padding: .3rem .6rem .3rem 0;
  border-bottom: 1px solid var(--line); }
td { padding: .32rem .6rem .32rem 0; border-bottom: 1px solid var(--line);
  vertical-align: top; }
td.num, th.num { text-align: right; padding-right: 0; }
pre { background: #8881; padding: .8rem; overflow-x: auto; border-radius: 4px;
  font-size: .85rem; }
.empty { color: var(--dim); padding: 1rem 0; }
.tag { display: inline-block; padding: .05rem .4rem; border: 1px solid var(--line);
  border-radius: 3px; font-size: .75rem; color: var(--dim); }
nav.crumbs { margin-bottom: 1rem; font-size: .85rem; }
"""


def layout(title: str, body: str, *, crumbs: Sequence[tuple[str, str]] = ()) -> str:
    """One page. ``title`` and every crumb label are escaped."""
    trail = " / ".join(
        f'<a href="{e(href)}">{e(label)}</a>' if href else e(label) for label, href in crumbs
    )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)} · Ledger</title>
<style>{STYLE}</style>
</head><body>
<header><div class="inner"><b><a href="/">Ledger</a></b>
<span class="dim">a version control system for RL environments</span></div></header>
<main>
{f'<nav class="crumbs">{trail}</nav>' if trail else ""}
{body}
</main></body></html>"""


@final
@dataclass(frozen=True, slots=True)
class Column:
    header: str
    numeric: bool = False


def column(header: str, *, numeric: bool = False) -> Column:
    return Column(header=header, numeric=numeric)


def table(columns: Sequence[Column], rows: Iterable[Sequence[object]], *, empty: str) -> str:
    """A table, or a note saying why there is none.

    Cells may be ``Markup`` where a link is needed; everything else is escaped.
    """
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{e(cell)}</td>' if col.numeric else f"<td>{e(cell)}</td>"
            for col, cell in zip(columns, row, strict=True)
        )
        + "</tr>"
        for row in rows
    )
    if not body:
        return f'<p class="empty">{e(empty)}</p>'
    head = "".join(
        f'<th class="num">{e(c.header)}</th>' if c.numeric else f"<th>{e(c.header)}</th>"
        for c in columns
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def link(label: object, href: str) -> Markup:
    """A link whose label and target are both escaped."""
    return Markup(f'<a href="{e(href)}">{e(label)}</a>')


def short(name: object) -> str:
    text = str(name)
    return text[:15] + "…" if len(text) > 16 else text
