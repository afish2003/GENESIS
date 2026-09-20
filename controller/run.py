"""Run assembly — the single place a run is wired together.

This existed inline in `main.py`, which meant nothing could exercise it except
launching the process. That is how `materialise_world_template` came to be
imported and never called: prompts were rendered for the run's dimensions while
identity statements came from whatever the previous run left on disk, producing
a run whose config.json and world disagreed, with no error.

Both `controller.main` and the integration tests now go through `prepare_run`,
so the wiring is covered by tests rather than by hand.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from controller.config import RunConfig
from controller.inference.backend import InferenceBackend
from controller.inference.factory import create_backend, create_evaluator_backend
from controller.logging.logger import AppendOnlyJSONLLogger
from controller.prompts import materialise_prompts, materialise_world_template
from controller.retrieval.databases import KnowledgeBaseManager
from controller.sandbox import ExecutionSandbox, create_sandbox
from controller.scenarios.library import load_scenario_library
from controller.world.reset import initialize_world, load_checkpoint
from controller.world.state import WorldState

logger = logging.getLogger(__name__)

#: Keys redacted from the config written into a run's log directory. Run
#: directories are what researchers archive and share; a live bearer token has
#: no business in one.
_SECRET_FIELDS = ("api_key", "evaluator_api_key")


def _provide(override: Optional[Path], dest: Path, render) -> Path:
    """Populate `dest` either by copying a verbatim override or by rendering.

    A researcher-supplied directory is copied, never rendered into and never
    deleted — `render_dir` starts with `shutil.rmtree(dest)`, which would
    otherwise destroy it.
    """
    if override is not None:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(override, dest)
        logger.info("Using %s verbatim -> %s", override, dest)
    else:
        render(dest)
    return dest


class CheckpointMismatchError(RuntimeError):
    """The world on disk is not the world the checkpoint was written for."""


def _verify_checkpoint(world: WorldState, checkpoint: dict) -> None:
    """Refuse to resume onto a world that does not match the checkpoint.

    Artifact writes are non-atomic, so a crash mid-save leaves a torn world
    directory. The checkpoint hash was previously recorded and never compared,
    so a resume continued against it silently.
    """
    expected = checkpoint.get("world_state_hash")
    if not expected:
        logger.warning("Checkpoint carries no world hash; cannot verify the world")
        return

    world.load()
    actual = world.compute_hash()
    if actual != expected:
        raise CheckpointMismatchError(
            f"Refusing to resume: the world directory does not match the "
            f"checkpoint for cycle {checkpoint.get('last_completed_cycle')}.\n"
            f"  checkpoint hash: {expected}\n"
            f"  world on disk:   {actual}\n"
            f"The last cycle probably crashed part-way through persisting. "
            f"Restore {world.world_dir} from the run's world_archive, or start "
            f"a fresh run."
        )
    logger.info("Checkpoint verified against the world on disk (%s)", actual)


@dataclass
class PreparedRun:
    config: RunConfig
    backend: InferenceBackend
    world: WorldState
    log: AppendOnlyJSONLLogger
    scenario_library: dict
    kb_manager: KnowledgeBaseManager
    sandbox: ExecutionSandbox
    evaluator_backend: Optional[InferenceBackend]
    start_cycle: int


def redact_config(config: RunConfig) -> dict:
    """Serialise a config for the run log with secrets removed."""
    data = config.model_dump(mode="json")
    for key in _SECRET_FIELDS:
        if data.get(key):
            data[key] = "<redacted>"
    return data


def prepare_run(
    config: RunConfig,
    *,
    resume: bool = False,
    load_embeddings: bool = True,
) -> PreparedRun:
    """Assemble everything a run needs, in the order it must happen."""
    log = AppendOnlyJSONLLogger(config.run_log_dir)
    log.initialize()
    log.write_config(redact_config(config))

    # Render BOTH sources against this run's dimensions, into this run's own
    # directory. Rendering only one of them is how prompts and identity
    # statements come to disagree; rendering into a shared directory is how
    # concurrent arms overwrite each other and how a researcher's hand-written
    # prompt directory got deleted.
    _provide(config.prompts_dir, config.run_prompts_dir,
             lambda dest: materialise_prompts(config, dest))
    _provide(config.world_template_dir, config.run_world_template_dir,
             lambda dest: materialise_world_template(config, dest))

    start_cycle = 0
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(config.run_log_dir)
        if checkpoint:
            start_cycle = checkpoint["last_completed_cycle"] + 1
            logger.info("Resuming from cycle %d", start_cycle)
        else:
            logger.warning("No checkpoint found; starting from cycle 0")

    if start_cycle == 0:
        initialize_world(config.run_world_template_dir, config.world_dir)

    world = WorldState(config.world_dir, agents=config.agents)

    if checkpoint and start_cycle > 0:
        _verify_checkpoint(world, checkpoint)

    kb_manager = KnowledgeBaseManager(
        kb_dir=config.knowledge_bases_dir,
        bm25_pool_size=config.bm25_candidate_pool,
        rerank_top_k=config.rerank_top_k,
        embedding_model=config.embedding_model,
    )
    kb_manager.initialize(load_embeddings=load_embeddings)

    # NullSandbox unless the researcher configured otherwise. Built once per run
    # rather than per phase; whether it actually works is checked by the
    # orchestrator at RUN_START, where the answer can be awaited and logged.
    sandbox = create_sandbox(config)
    if config.execution_enabled:
        logger.info(
            "Execution is ENABLED: backend=%s, image=%s, %.0fs per run",
            config.sandbox_backend.value, config.sandbox_image,
            config.sandbox_timeout_seconds,
        )

    return PreparedRun(
        config=config,
        backend=create_backend(config),
        world=world,
        log=log,
        scenario_library=load_scenario_library(config),
        kb_manager=kb_manager,
        sandbox=sandbox,
        evaluator_backend=create_evaluator_backend(config),
        start_cycle=start_cycle,
    )

