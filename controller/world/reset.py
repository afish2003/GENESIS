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


def initialize_world(world_template_dir: Path, world_dir: Path) -> None:
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
