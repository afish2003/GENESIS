"""Filesystem containment for model-supplied identifiers.

GENESIS has no OS-level sandbox: the controller runs as the researcher's own
user, and agent output is untrusted input that reaches `pathlib` operations.
Anything an agent names must therefore be treated as hostile before it is used
to build a path.

Two vectors matter, and pathlib defends against neither:

    protocols_dir / "../../../etc/foo"   -> traverses upward
    protocols_dir / "/etc/foo"           -> DISCARDS the base entirely

The second is the sharper one: when the right operand is absolute, `/` throws
the base away, so no ".." is needed to escape.

Defence is layered. `safe_artifact_id()` sanitises at ingress, so the stored
identifier is already safe and stays consistent across world state and logs.
`assert_within()` is the hard invariant at egress — it fires if any future
code path builds a write target from untrusted data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Conservative allowlist. Anything outside it becomes "_".
_ALLOWED = re.compile(r"[^A-Za-z0-9._-]")

# Reserved device names on Windows. The study runs on macOS/Linux, but logs and
# world directories get copied between machines, so refuse them cheaply.
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

MAX_ID_LENGTH = 64


class SandboxEscapeError(RuntimeError):
    """A path resolved outside the sealed world directory."""


def safe_artifact_id(raw: str, fallback: str = "unnamed_protocol") -> str:
    """Reduce a model-supplied identifier to a single safe filename component.

    Guarantees about the return value: it contains no path separator, is not
    "." or "..", does not start with a dot, is non-empty, and is at most
    MAX_ID_LENGTH characters.

    >>> safe_artifact_id("PROTO_001")
    'PROTO_001'
    >>> safe_artifact_id("../../etc/passwd")
    'etc_passwd'
    >>> safe_artifact_id("/absolute/path")
    'absolute_path'
    """
    if not isinstance(raw, str):
        return fallback

    # Normalise separators first so traversal segments are visible as text.
    candidate = raw.strip().replace("\\", "/")

    # Drop empty and dot-only segments — this is what removes "..".
    segments = [s for s in candidate.split("/") if s and s.strip(".") != ""]
    candidate = "_".join(segments)

    candidate = candidate.replace("\x00", "")
    candidate = _ALLOWED.sub("_", candidate)
    candidate = candidate.strip("._-")
    candidate = re.sub(r"_{2,}", "_", candidate)
    candidate = candidate[:MAX_ID_LENGTH].strip("._-")

    if not candidate or candidate.lower() in _RESERVED:
        logger.warning(
            "Model-supplied identifier %r sanitised to nothing; using %r", raw, fallback
        )
        return fallback

    if candidate != raw:
        logger.info("Sanitised artifact id %r -> %r", raw, candidate)

    return candidate


def assert_within(base: Path, target: Path) -> Path:
    """Return `target` resolved, or raise if it escapes `base`.

    The last line of defence before a write. Resolves symlinks, so a symlink
    planted inside the world directory cannot redirect a write outside it.
    """
    base_resolved = Path(base).resolve()
    target_resolved = Path(target).resolve()

    try:
        target_resolved.relative_to(base_resolved)
    except ValueError:
        raise SandboxEscapeError(
            f"Refusing to write outside the sealed world.\n"
            f"  base:   {base_resolved}\n"
            f"  target: {target_resolved}"
        ) from None

    return target_resolved
