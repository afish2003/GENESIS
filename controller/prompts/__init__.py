"""Prompt and world-template resolution.

Single source, composable dimensions. `prompts_src/` and `world_template_src/`
hold one copy of each file with its variation expressed inline; a run renders
them against its own dimensions into a materialised directory.

This replaces the earlier prompts/ + prompts_undisclosed/ + prompts_minimal/
arrangement, in which every new dimension doubled the number of directories and
the copies drifted apart silently.
"""

from controller.prompts.render import (
    PromptRenderError,
    render,
    render_dir,
    render_file,
)
from controller.prompts.resolve import (
    DIMENSIONS,
    materialise_prompts,
    materialise_world_template,
    render_dimensions,
)

__all__ = [
    "render",
    "render_file",
    "render_dir",
    "PromptRenderError",
    "DIMENSIONS",
    "render_dimensions",
    "materialise_prompts",
    "materialise_world_template",
]
