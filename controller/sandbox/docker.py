"""DockerSandbox — hardened container execution.

Implements the properties specified in docs/containment_design.md. Every flag
below is load-bearing; the table in that document says what each one defends
against. Notably absent, and deliberately so: any Docker socket mount, any
--privileged, any capability, and any mount other than the workspace.

Threat model is accident and prompt-space pressure, not a kernel 0-day. For
stronger isolation the runtime can be pointed at gVisor (`--runtime runsc`)
without changing anything else here.

NOT YET WIRED INTO THE CYCLE LOOP. Nothing in the current experimental design
requires code execution, and enabling it is gated on the precondition
checklist in the design document — including an escape-test suite that must
pass on the actual host.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path

from controller.sandbox.base import MAX_TIMEOUT_SECONDS, ExecutionSandbox
from controller.sandbox.schemas import ExecutionOutcome, ExecutionRequest, ExecutionResult
from controller.world.paths import safe_artifact_id

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "python:3.11-slim"

# OCI exit codes
_EXIT_TIMEOUT_KILL = 137  # SIGKILL — also what the OOM killer produces


class DockerSandbox(ExecutionSandbox):
    """Runs agent code in a throwaway, network-less, non-root container."""

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        runtime: str = "docker",          # or "podman", or "docker --runtime runsc"
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 128,
        workspace_root: Path | None = None,
    ) -> None:
        self.image = image
        self.runtime = runtime
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.workspace_root = workspace_root

    @property
    def runtime_argv(self) -> list[str]:
        """The runtime as argv.

        `runtime` is documented as accepting "docker --runtime runsc" for
        gVisor, and health_check split it — but _container_args passed the whole
        string as a single argv element to create_subprocess_exec, so the
        documented gVisor path looked up an executable literally named
        "docker --runtime runsc" and failed with FileNotFoundError.
        """
        return self.runtime.split()

    def _container_args(self, workspace: Path, timeout: float) -> list[str]:
        """Every flag here is a containment property, not a preference."""
        return [
            *self.runtime_argv, "run",
            "--rm",                         # ephemeral: no cross-execution persistence
            "--network", "none",            # no exfiltration, no fetching, no LAN, no Ollama
            "--read-only",                  # immutable rootfs
            "--tmpfs", "/tmp:size=64m,noexec,nosuid",
            "--user", "65534:65534",        # nobody
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", self.memory,
            "--memory-swap", self.memory,   # equal to memory => swap disabled
            "--cpus", self.cpus,
            "--pids-limit", str(self.pids_limit),
            # The ONLY mount. Never the repo, never .env, never world/ above this.
            "-v", f"{workspace}:/workspace:rw",
            "-w", "/workspace",
            self.image,
            "timeout", "--signal=KILL", str(int(timeout)),
            "sh", "-c", "",  # replaced by the caller
        ]

    async def run(self, request: ExecutionRequest) -> ExecutionResult:
        timeout = min(request.timeout_seconds, MAX_TIMEOUT_SECONDS)
        workspace = Path(tempfile.mkdtemp(dir=self.workspace_root, prefix="genesis_ws_"))
        started = time.monotonic()

        try:
            self._materialise(request, workspace)

            container_name = f"genesis_{workspace.name}"
            args = self._container_args(workspace, timeout)
            args.insert(2, "--name")
            args.insert(3, container_name)
            args[-1] = request.entrypoint

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                # Outer deadline in case the container runtime itself wedges.
                raw_out, raw_err = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout + 15
                )
            except asyncio.TimeoutError:
                await self._kill(proc, container_name)
                return ExecutionResult(
                    outcome=ExecutionOutcome.TIMEOUT,
                    duration_seconds=time.monotonic() - started,
                    detail=f"Container did not exit within {timeout + 15:.0f}s; killed.",
                )

            stdout, t1 = self.truncate(raw_out.decode("utf-8", errors="replace"))
            stderr, t2 = self.truncate(raw_err.decode("utf-8", errors="replace"))
            elapsed = time.monotonic() - started

            outcome = ExecutionOutcome.OK
            detail = ""
            if proc.returncode == _EXIT_TIMEOUT_KILL:
                # SIGKILL: the timeout fired, or the OOM killer did.
                outcome = ExecutionOutcome.TIMEOUT
                detail = f"Killed after {timeout:.0f}s, or exceeded {self.memory} memory."
            elif proc.returncode != 0:
                outcome = ExecutionOutcome.NONZERO_EXIT

            return ExecutionResult(
                outcome=outcome,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=elapsed,
                truncated=t1 or t2,
                detail=detail,
            )

        except FileNotFoundError:
            return ExecutionResult(
                outcome=ExecutionOutcome.SANDBOX_ERROR,
                detail=f"Container runtime {self.runtime!r} not found on PATH.",
            )
        except Exception as e:
            logger.exception("Sandbox failure")
            return ExecutionResult(
                outcome=ExecutionOutcome.SANDBOX_ERROR,
                detail=f"{type(e).__name__}: {e}",
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def _materialise(self, request: ExecutionRequest, workspace: Path) -> None:
        """Write requested files into the workspace, flattening any path games.

        Agent-supplied filenames get the same treatment as protocol_id: reduced
        to a single safe component, so nothing can be written outside the
        workspace even before the container's own isolation applies.
        """
        for raw_name, content in request.files.items():
            name = safe_artifact_id(raw_name, fallback="file.txt")
            (workspace / name).write_text(content, encoding="utf-8")

    async def _kill(self, proc, container: str | None = None) -> None:
        """Kill the container, then the client.

        Killing only the client left the container running: --rm cleans up when
        the container exits, not when the client dies, so the "outer deadline in
        case the runtime wedges" did not actually stop a wedged container.
        """
        if container:
            try:
                killer = await asyncio.create_subprocess_exec(
                    *self.runtime_argv, "kill", container,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(killer.wait(), timeout=15)
            except (OSError, asyncio.TimeoutError) as e:
                logger.error("Could not kill container %s: %s", container, e)
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass

    async def health_check(self) -> bool:
        if shutil.which(self.runtime_argv[0]) is None:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                self.runtime_argv[0], "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=15)
            return proc.returncode == 0
        except (asyncio.TimeoutError, OSError):
            return False
