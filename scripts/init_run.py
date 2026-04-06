"""Initialize a new experimental run.

Creates the run log directory, initializes the world from template,
copies prompts for version-locking, and writes the run config.

Usage:
    python scripts/init_run.py --run-id RUN_001 --condition BASELINE --cycles 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.config import load_config
from controller.logging.logger import AppendOnlyJSONLLogger
from controller.world.reset import archive_world, initialize_world


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a GENESIS run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", required=True, choices=["BASELINE", "MEM_RESET"])
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--archive-previous",
        type=str,
        default=None,
        help="Run ID of previous run to archive world state from",
    )
    args = parser.parse_args()

    config = load_config(
        run_id=args.run_id,
        condition=args.condition,
        cycles=args.cycles,
        config_file=args.config,
    )

    # Archive previous world state if requested
    if args.archive_previous:
        prev_log_dir = config.research_logs_dir / args.archive_previous
        if prev_log_dir.exists():
            archive_world(config.world_dir, prev_log_dir)
            print(f"Archived world state to {prev_log_dir}/world_archive/")
        else:
            print(f"Warning: Previous run directory {prev_log_dir} not found, skipping archive")

    # Initialize world from template
    initialize_world(config.world_template_dir, config.world_dir)
    print(f"Initialized world from template: {config.world_template_dir} -> {config.world_dir}")

    # Initialize log directory
    log = AppendOnlyJSONLLogger(config.run_log_dir)
    log.initialize()
    print(f"Created run log directory: {config.run_log_dir}")

    # Write config
    log.write_config(config.model_dump(mode="json"))
    print(f"Wrote config.json")

    # Copy prompts for version-locking
    log.copy_prompts(config.prompts_dir)
    print(f"Copied prompts for version-locking")

    print(f"\nRun {config.run_id} initialized. Ready to execute:")
    print(f"  python -m controller.main --run-id {config.run_id} --condition {config.condition.value} --cycles {config.total_cycles}")


if __name__ == "__main__":
    main()
