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
from controller.inference.factory import create_backend
from controller.logging.logger import AppendOnlyJSONLLogger
from controller.prompts import materialise_prompts, materialise_world_template
from controller.retrieval.databases import KnowledgeBaseManager
from controller.scenarios.library import load_scenario_library
from controller.world.reset import initialize_world, load_checkpoint
from controller.world.state import WorldState

logger = logging.getLogger(__name__)

#: Keys redacted from the config written into a run's log directory. Run
#: directories are what researchers archive and share; a live bearer token has
#: no business in one.
_SECRET_FIELDS = ("api_key",)


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


@dataclass
class PreparedRun:
    config: RunConfig
    backend: InferenceBackend
    world: WorldState
    log: AppendOnlyJSONLLogger
    scenario_library: dict
    kb_manager: KnowledgeBaseManager
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

    kb_manager = KnowledgeBaseManager(
        kb_dir=config.knowledge_bases_dir,
        bm25_pool_size=config.bm25_candidate_pool,
        rerank_top_k=config.rerank_top_k,
        embedding_model=config.embedding_model,
    )
    kb_manager.initialize(load_embeddings=load_embeddings)

    return PreparedRun(
        config=config,
        backend=create_backend(config),
        world=world,
        log=log,
        scenario_library=load_scenario_library(config),
        kb_manager=kb_manager,
        start_cycle=start_cycle,
    )
