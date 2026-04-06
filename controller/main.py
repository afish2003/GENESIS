"""GENESIS controller entry point.

Usage:
    python -m controller.main --run-id RUN_001 --condition BASELINE --cycles 100
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

from controller.config import load_config

console = Console()


def setup_logging() -> None:
    """Configure logging with rich handler."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="genesis",
        description="GENESIS multi-agent research experiment controller",
    )
    parser.add_argument("--run-id", required=True, help="Unique run identifier (e.g. RUN_001)")
    parser.add_argument(
        "--condition",
        required=True,
        choices=["BASELINE", "MEM_RESET"],
        help="Experimental condition",
    )
    parser.add_argument("--cycles", type=int, default=100, help="Number of cycles to run")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file")
    parser.add_argument(
        "--pause-after-cycle",
        type=int,
        default=None,
        help="Pause for human inspection after this cycle",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from last checkpoint if available",
    )
    return parser.parse_args(argv)


async def run(argv: list[str] | None = None) -> None:
    """Async entry point for the controller."""
    args = parse_args(argv)
    setup_logging()

    config = load_config(
        run_id=args.run_id,
        condition=args.condition,
        cycles=args.cycles,
        config_file=args.config,
        pause_after_cycle=args.pause_after_cycle,
    )

    console.print(f"[bold green]GENESIS[/] — Run [bold]{config.run_id}[/]")
    console.print(f"  Condition: {config.condition.value}")
    console.print(f"  Cycles:    {config.total_cycles}")
    console.print(f"  Model:     {config.model_name}")
    console.print(f"  Ollama:    {config.ollama_host}")
    console.print()

    # Initialize inference backend
    from controller.inference.ollama_backend import OllamaBackend

    backend = OllamaBackend(
        host=config.ollama_host,
        model=config.model_name,
    )

    # Initialize world state
    from controller.world.reset import initialize_world, load_checkpoint
    from controller.world.state import WorldState

    # Initialize logging
    from controller.logging.logger import AppendOnlyJSONLLogger

    log = AppendOnlyJSONLLogger(config.run_log_dir)
    log.initialize()

    # Write run config to log directory
    log.write_config(config.model_dump(mode="json"))

    # Copy prompts for version-locking
    log.copy_prompts(config.prompts_dir)

    # Determine start cycle
    start_cycle = 0
    if args.resume:
        checkpoint = load_checkpoint(config.run_log_dir)
        if checkpoint:
            start_cycle = checkpoint["last_completed_cycle"] + 1
            console.print(f"[yellow]Resuming from cycle {start_cycle}[/]")
        else:
            console.print("[yellow]No checkpoint found, starting from cycle 0[/]")

    # Initialize world from template (only if starting fresh)
    if start_cycle == 0:
        initialize_world(config.world_template_dir, config.world_dir)

    world = WorldState(config.world_dir)

    # Load scenario library
    from controller.scenarios.library import load_scenario_library

    scenario_library = load_scenario_library(config)

    # Initialize knowledge bases
    from controller.retrieval.databases import KnowledgeBaseManager

    kb_manager = KnowledgeBaseManager(
        kb_dir=config.knowledge_bases_dir,
        bm25_pool_size=config.bm25_candidate_pool,
        rerank_top_k=config.rerank_top_k,
        embedding_model=config.embedding_model,
    )
    kb_manager.initialize(load_embeddings=True)

    # Build and run the cycle orchestrator
    from controller.cycle import CycleOrchestrator

    orchestrator = CycleOrchestrator(
        config=config,
        backend=backend,
        world=world,
        log=log,
        scenario_library=scenario_library,
        kb_manager=kb_manager,
    )

    try:
        await orchestrator.run_all_cycles(start_cycle=start_cycle)
        console.print(f"\n[bold green]Run {config.run_id} complete.[/]")
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Run {config.run_id} interrupted.[/]")
    finally:
        await backend.close()


def main(argv: list[str] | None = None) -> None:
    """Synchronous entry point."""
    asyncio.run(run(argv))


if __name__ == "__main__":
    main()


def cli_entry() -> None:
    """Entry point for console_scripts."""
    main()
