"""Between-run reset: archive current world state and reinitialize from template.

The world_template/ directory is the canonical clean state. It is never
modified during runs. Before each new run, the working world/ directory
is archived into the previous run's log directory and then replaced
with a fresh copy from the template.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def archive_world(world_dir: Path, archive_dest: Path) -> None:
    """Copy the current world directory into the run's archive location."""
    if not world_dir.exists():
        logger.warning("No world directory to archive at %s", world_dir)
        return

    archive_dest.mkdir(parents=True, exist_ok=True)
    target = archive_dest / "world_archive"

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(world_dir, target)
    logger.info("Archived world state to %s", target)


def initialize_world(world_template_dir: Path, world_dir: Path, config=None) -> None:
    """Initialize (or reinitialize) the working world from the clean template.

    If the world directory already exists, it is fully replaced.
    """
    if not world_template_dir.exists():
        raise FileNotFoundError(
            f"World template directory not found: {world_template_dir}"
        )

    if world_dir.exists():
        shutil.rmtree(world_dir)

    shutil.copytree(world_template_dir, world_dir)
    logger.info("Initialized world from template: %s -> %s", world_template_dir, world_dir)

    if config is not None:
        _materialise_missing_identities(world_dir, config)


#: Rendered per run alongside the rest of the template; filled in per agent here.
GENERIC_IDENTITY = "identity_agent.md"


def _materialise_missing_identities(world_dir: Path, config) -> None:
    """Give every roster agent an identity statement.

    A hand-written identity_<id>.md wins. Anything else is written from the
    generic template with the roster substituted in, so `agents: [a, b, c, d]`
    needs no files — which is the whole point of the roster being config.

    Without this an agent with no file got no identity at all: _load_identities
    skips a missing file silently, and the first thing to touch
    world.identities[agent_id] raises a KeyError several phases later.
    """
    from controller.agents.base import fill_roster

    doctrine_dir = world_dir / "doctrine"
    template_path = doctrine_dir / GENERIC_IDENTITY
    for agent_id in config.agents:
        target = doctrine_dir / f"identity_{agent_id}.md"
        if target.exists():
            continue
        if not template_path.exists():
            raise FileNotFoundError(
                f"{agent_id} has no identity_{agent_id}.md and the generic "
                f"template {template_path} is missing — the agent would start "
                f"with no identity statement at all."
            )
        target.write_text(
            fill_roster(template_path.read_text(encoding="utf-8"), agent_id, config),
            encoding="utf-8",
        )
        logger.info("Wrote %s from the generic identity template", target.name)

    # The template itself is not one of the agents' documents.
    template_path.unlink(missing_ok=True)


def write_checkpoint(run_log_dir: Path, run_id: str, cycle: int, world_hash: str) -> None:
    """Write a checkpoint file after a cycle completes."""
    checkpoint = {
        "run_id": run_id,
        "last_completed_cycle": cycle,
        "world_state_hash": world_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    filepath = run_log_dir / "checkpoint.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(run_log_dir: Path) -> dict | None:
    """Load checkpoint if it exists. Returns None if no checkpoint found."""
    filepath = run_log_dir / "checkpoint.json"
    if not filepath.exists():
        return None
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)
