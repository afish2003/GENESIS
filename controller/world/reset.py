"""World archive, initialization, and checkpoint utilities.

Used by init_run.py between experimental runs, and by the cycle
orchestrator to checkpoint progress for run resumption.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def initialize_world(template_dir: Path, world_dir: Path) -> None:
    """Copy the world_template into world_dir, replacing any existing content.

    If world_dir already exists it is removed first so the copy is clean.
    """
    template_dir = Path(template_dir)
    world_dir = Path(world_dir)

    if not template_dir.exists():
        raise FileNotFoundError(f"World template not found: {template_dir}")

    if world_dir.exists():
        shutil.rmtree(world_dir)

    shutil.copytree(template_dir, world_dir)
    logger.info("Initialized world at %s from template %s", world_dir, template_dir)


def archive_world(world_dir: Path, archive_dir: Path) -> None:
    """Copy the current world state into archive_dir/world_archive/.

    Used before a between-run reset so the previous state is preserved.
    """
    world_dir = Path(world_dir)
    archive_dir = Path(archive_dir)

    dest = archive_dir / "world_archive"
    if dest.exists():
        shutil.rmtree(dest)

    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(world_dir, dest)
    logger.info("Archived world from %s to %s", world_dir, dest)


def write_checkpoint(
    run_log_dir: Path,
    run_id: str,
    last_completed_cycle: int,
    world_hash: str,
) -> None:
    """Write a JSON checkpoint file so interrupted runs can resume.

    The checkpoint records the last successfully completed cycle and the
    world hash at that point.
    """
    run_log_dir = Path(run_log_dir)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "run_id": run_id,
        "last_completed_cycle": last_completed_cycle,
        "world_hash": world_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path = run_log_dir / "checkpoint.json"
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    logger.debug("Checkpoint written: cycle %d, hash %s", last_completed_cycle, world_hash)


def load_checkpoint(run_log_dir: Path) -> Optional[dict]:
    """Load the checkpoint file from run_log_dir, or return None if absent."""
    path = Path(run_log_dir) / "checkpoint.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read checkpoint at %s: %s", path, exc)
        return None
