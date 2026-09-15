"""Initialize a run: render prompts and world template, create the log directory.

Thin wrapper over controller.run.prepare_run. It used to assemble a run itself,
in a slightly different order from controller/main.py — which is how main.py
came to render prompts but not the world template, producing runs whose
config.json and identity statements disagreed. One code path now, two entry
points.

Usage:
    python scripts/init_run.py --run-id RUN_001 --condition BASELINE --cycles 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.config import load_config
from controller.run import prepare_run
from controller.world.reset import archive_world


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a GENESIS run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", required=True, choices=["BASELINE", "MEM_RESET"])
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--framing", choices=["disclosed", "undisclosed"], default=None)
    parser.add_argument("--identity-seed", choices=["prescribed", "minimal"], default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--archive-previous",
        type=str,
        default=None,
        help="Run id whose log directory should receive the current world state",
    )
    args = parser.parse_args()

    config = load_config(
        run_id=args.run_id,
        condition=args.condition,
        cycles=args.cycles,
        config_file=args.config,
        framing=args.framing,
        identity_seed=args.identity_seed,
    )

    if args.archive_previous:
        prev_log_dir = config.research_logs_dir / args.archive_previous
        if prev_log_dir.exists():
            archive_world(config.world_dir, prev_log_dir)
            print(f"Archived world state to {prev_log_dir}/world_archive/")
        else:
            print(f"Warning: {prev_log_dir} not found, skipping archive")

    # Embeddings are only needed to serve queries, not to initialise a run.
    prepare_run(config, load_embeddings=False)

    print(f"Run log directory:  {config.run_log_dir}")
    print(f"Prompts rendered:   {config.run_prompts_dir} "
          f"(framing={config.framing.value}, identity_seed={config.identity_seed.value})")
    print(f"World initialized:  {config.world_dir}")
    print(f"Agents:             {', '.join(config.agents)}")
    print(f"Task:               {config.task}")
    print(f"\nRun {config.run_id} initialized. Ready to execute:")
    print(f"  python -m controller.main --run-id {config.run_id} "
          f"--condition {config.condition.value} --cycles {config.total_cycles}")


if __name__ == "__main__":
    main()
