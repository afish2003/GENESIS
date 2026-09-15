"""Turn a RunConfig's dimensions into concrete prompt and world directories."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from controller.prompts.render import render_file

if TYPE_CHECKING:
    from controller.config import RunConfig

logger = logging.getLogger(__name__)

# Every dimension a prompt source may condition on. Adding one here is the only
# change needed to make it available to `<!--IF name=value-->` blocks — no new
# directories, no copies.
DIMENSIONS = ("framing", "identity_seed")


def render_dimensions(config: RunConfig) -> dict[str, str]:
    """The dimension values this run renders with."""
    return {
        "framing": config.framing.value,
        "identity_seed": config.identity_seed.value,
    }


def _materialise(src: Path, dest: Path, dimensions: dict[str, str]) -> Path:
    """Render every .md under `src` into `dest`, copying other files verbatim."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        out = dest / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".md":
            out.write_text(render_file(path, dimensions), encoding="utf-8")
        else:
            shutil.copy2(path, out)

    # Preserve empty directories the world layout depends on (sandbox/protocols).
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            (dest / path.relative_to(src)).mkdir(parents=True, exist_ok=True)

    return dest


def materialise_prompts(config: RunConfig, dest: Path) -> Path:
    """Render the prompt set for this run and return the directory."""
    dims = render_dimensions(config)
    out = _materialise(config.prompts_src_dir, dest, dims)
    logger.info("Rendered prompts (%s) -> %s",
                ", ".join(f"{k}={v}" for k, v in dims.items()), out)
    return out


def materialise_world_template(config: RunConfig, dest: Path) -> Path:
    """Render the clean world template for this run and return the directory."""
    dims = render_dimensions(config)
    out = _materialise(config.world_template_src_dir, dest, dims)
    logger.info("Rendered world template (%s) -> %s",
                ", ".join(f"{k}={v}" for k, v in dims.items()), out)
    return out
