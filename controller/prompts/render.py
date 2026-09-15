"""Conditional rendering for prompt sources.

Prompt variants used to be whole duplicated directories — prompts/,
prompts_undisclosed/, prompts_minimal/. That does not compose: two dimensions
need four directories, three need eight, and the copies drift silently. It also
produced a real bug, where selecting a framing was ignored without warning
because an explicit prompts_dir won.

Instead each prompt is a single source file carrying its own variation inline:

    You are Axiom<!--IF framing=disclosed-->, one of two AI agents operating
    within GENESIS — a contained research environment<!--ENDIF-->.

Dimensions are supplied as a plain dict (framing, identity_seed, ...), and the
rendered result is materialised into the run's log directory, so what was
actually sent is still version-locked per run exactly as before.

Syntax, deliberately minimal:

    <!--IF key=value-->kept when key == value<!--ENDIF-->
    <!--IF key=value-->kept<!--ELSE-->kept otherwise<!--ENDIF-->

No nesting, no expressions. An unknown key, an unclosed block or a stray ELSE
raises rather than leaking markup into a prompt an agent would then read.
"""

from __future__ import annotations

import re
from pathlib import Path

_BLOCK = re.compile(
    r"<!--IF\s+(?P<key>[a-z_]+)\s*=\s*(?P<value>[A-Za-z0-9_\-]+)\s*-->"
    r"(?P<then>.*?)"
    r"(?:<!--ELSE-->(?P<otherwise>.*?))?"
    r"<!--ENDIF-->",
    re.DOTALL,
)

_STRAY = re.compile(r"<!--\s*(IF|ELSE|ENDIF)\b")


class PromptRenderError(ValueError):
    """A prompt source could not be rendered. Never silently degraded."""


def render(source: str, dimensions: dict[str, str], *, origin: str = "<string>") -> str:
    """Resolve every conditional block in `source` against `dimensions`."""

    def replace(match: re.Match) -> str:
        key = match.group("key")
        if key not in dimensions:
            raise PromptRenderError(
                f"{origin}: prompt references unknown dimension {key!r}. "
                f"Known dimensions: {sorted(dimensions)}"
            )
        taken = match.group("then") if dimensions[key] == match.group("value") else (
            match.group("otherwise") or ""
        )
        return taken

    rendered = _BLOCK.sub(replace, source)

    leftover = _STRAY.search(rendered)
    if leftover:
        line = rendered[: leftover.start()].count("\n") + 1
        raise PromptRenderError(
            f"{origin}: unbalanced conditional near line {line} — "
            f"found {leftover.group(0)!r} with no matching block. "
            f"Refusing to emit a prompt containing template markup."
        )

    return rendered


def render_file(path: Path, dimensions: dict[str, str]) -> str:
    return render(path.read_text(encoding="utf-8"), dimensions, origin=str(path))


def render_dir(src: Path, dest: Path, dimensions: dict[str, str]) -> list[Path]:
    """Render every .md in `src` into `dest`. Returns the files written."""
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(src.glob("*.md")):
        out = dest / path.name
        out.write_text(render_file(path, dimensions), encoding="utf-8")
        written.append(out)
    return written
