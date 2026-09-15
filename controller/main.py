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
from controller.inference.factory import describe_backend
from controller.run import prepare_run

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
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai", "mock"],
        default=None,
        help="Inference backend (overrides INFERENCE_BACKEND in .env)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides MODEL_NAME/OLLAMA_MODEL in .env)",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible endpoint root incl. /v1 (overrides API_BASE_URL)",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file")
    parser.add_argument(
        "--framing",
        choices=["disclosed", "undisclosed"],
        default=None,
        help="Whether agents are told they are studied (overrides FRAMING in .env)",
    )
    parser.add_argument(
        "--identity-seed",
        choices=["prescribed", "minimal"],
        default=None,
        help="How much identity is given rather than developed",
    )
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
        framing=args.framing,
        identity_seed=args.identity_seed,
        inference_backend=args.backend,
        model_name=args.model,
        api_base_url=args.api_base_url,
    )

    console.print(f"[bold green]GENESIS[/] — Run [bold]{config.run_id}[/]")
    console.print(f"  Condition: {config.condition.value}")
    console.print(f"  Cycles:    {config.total_cycles}")
    console.print(f"  Model:     {config.model_name}")
    console.print(f"  Inference: {describe_backend(config)}")
    console.print(f"  Framing:   {config.framing.value} | seed: {config.identity_seed.value}")
    console.print()

    prepared = prepare_run(config, resume=args.resume)

    # Build and run the cycle orchestrator
    from controller.cycle import CycleOrchestrator

    orchestrator = CycleOrchestrator(
        config=prepared.config,
        backend=prepared.backend,
        world=prepared.world,
        log=prepared.log,
        scenario_library=prepared.scenario_library,
        kb_manager=prepared.kb_manager,
        sandbox=prepared.sandbox,
    )

    try:
        await orchestrator.run_all_cycles(start_cycle=prepared.start_cycle)
        console.print(f"\n[bold green]Run {config.run_id} complete.[/]")
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Run {config.run_id} interrupted.[/]")
    finally:
        await prepared.backend.close()
        await prepared.sandbox.close()


def main(argv: list[str] | None = None) -> None:
    """Synchronous entry point."""
    asyncio.run(run(argv))


if __name__ == "__main__":
    main()


def cli_entry() -> None:
    """Entry point for console_scripts."""
    main()
