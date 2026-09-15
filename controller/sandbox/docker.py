"""DockerSandbox — hardened container execution.

Implements the properties specified in docs/containment_design.md. Every flag
below is load-bearing; the table in that document says what each one defends
against. Notably absent, and deliberately so: any Docker socket mount, any
--privileged, any capability, and any mount other than the workspace.

Threat model is accident and prompt-space pressure, not a kernel 0-day. For
stronger isolation the runtime can be pointed at gVisor (`--runtime runsc`)
without changing anything else here.

Reachable from the cycle loop via the optional `execution` phase, behind two
separate switches: `execution_enabled` adds the phase, `sandbox_backend=docker`
selects this class over NullSandbox. Neither is on by default.

Before enabling it on a new host, run `scripts/verify_containment.py` — the
properties asserted in tests/test_sandbox.py are properties of the *command*,
and only that script checks they hold in an actual container on this machine.
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

#: Fraction of the runtime's total memory a single container may be allowed to
#: claim. The sandbox runs one container at a time, but the runtime's memory is
#: shared with everything else it hosts, and a --memory at or above the total is
#: not a limit at all — the cgroup can never be the thing that stops a runaway.
_MAX_MEMORY_FRACTION = 0.75


def parse_memory(value: str) -> int:
    """Docker's memory syntax ('512m', '1g', '1.5g', '2048') to bytes.

    Returns 0 for anything unparseable rather than raising: this feeds an
    advisory capacity check, and a config the sandbox cannot interpret is
    Docker's problem to reject, with a better message than ours.
    """
    text = str(value).strip().lower()
    units = {"b": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}
    multiplier = 1
    if text and text[-1] in units:
        multiplier = units[text[-1]]
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0

# OCI exit codes
_EXIT_TIMEOUT_KILL = 137  # SIGKILL — also what the OOM killer produces


class DockerSandbox(ExecutionSandbox):
    """Runs agent code in a throwaway, network-less, non-root container."""

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        runtime: str = "docker",          # or "podman", or "docker --runtime runsc"
        memory: str = "2g",
        cpus: str = "1.0",
        pids_limit: int = 128,
        workspace_root: Path | None = None,
        tmpfs: str | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self.image = image
        self.runtime = runtime
        self.memory = memory
        # /tmp is the container's only writable filesystem, and tmpfs pages are
        # charged to its memory cgroup — measured, not assumed: with
        # --memory 512m and --tmpfs size=1g, writing 900 MiB to /tmp is
        # OOM-killed at 137. So `size=` is not what contains a runaway write;
        # --memory is. Defaulting it below --memory would therefore be a second,
        # smaller, arbitrary limit that buys no containment and only takes
        # scratch space away from the agents. It tracks --memory instead, and
        # `sandbox_memory` is the one number to turn.
        self.tmpfs = tmpfs or memory
        # Persistent, writable, and the ONLY mount of either kind. Validated by
        # controller/sandbox/project.py before it ever reaches here: it must be
        # its own size-capped filesystem and must not be inside the repo, the
        # world directory or the logs. None is the default and means the agents
        # get nothing persistent.
        self.project_dir = project_dir
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
            # The only writable filesystem: RAM-backed, and bounded by the
            # memory cgroup rather than by this number. See __init__.
            "--tmpfs", f"/tmp:size={self.tmpfs},nosuid",
            "--user", "65534:65534",        # nobody
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", self.memory,
            "--memory-swap", self.memory,   # equal to memory => swap disabled
            "--cpus", self.cpus,
            "--pids-limit", str(self.pids_limit),
            *self._project_mount(),
            # This cycle's artifact. Read-only.
            #
            # It was `rw`, which left the disk-fill vector — the most likely
            # accident in the threat model — uncontained: --memory caps RAM and
            # --tmpfs caps /tmp, but a bind mount to the host filesystem has no
            # quota at all, so `open("big","w").write("x" * huge)` in a loop
            # filled the researcher's disk with every other limit intact.
            # Nothing reads files back out of the workspace — results are
            # stdout/stderr — so read-only costs nothing and closes it. Agent
            # code that needs scratch space uses /tmp, which is capped.
            "-v", f"{workspace}:/workspace:ro",
            # Working directory is the persistent project when there is one:
            # a program that builds a codebase has to run inside the codebase.
            # /workspace holds only this cycle's module, read-only, and the
            # entrypoint names it by absolute path.
            "-w", "/project" if self.project_dir else "/workspace",
            self.image,
            "timeout", "--signal=KILL", str(int(timeout)),
            "sh", "-c", "",  # replaced by the caller
        ]

    def _project_mount(self) -> list[str]:
        """The persistent project volume, if this run has one.

        Writable — which is only acceptable because it is a filesystem that is
        genuinely its configured size, so a runaway write ends in ENOSPC rather
        than in a full host disk. Docker cannot cap a bind mount; the kernel can
        cap a filesystem.
        """
        if self.project_dir is None:
            return []
        return ["-v", f"{self.project_dir}:/project:rw"]

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
                # The in-container `timeout` should have ended this already, so
                # reaching here means the runtime itself is not responding. That
                # is a containment event, not a slow program.
                killed = await self._kill(proc, container_name)
                return ExecutionResult(
                    outcome=ExecutionOutcome.TIMEOUT,
                    duration_seconds=time.monotonic() - started,
                    limit_hit=("runtime_unresponsive" if killed
                               else "container_may_still_be_running"),
                    detail=(
                        f"Container did not exit within {timeout + 15:.0f}s. "
                        + ("Killed." if killed else
                           f"THE KILL ALSO FAILED: container {container_name!r} "
                           f"may still be running. See the recovery procedure in "
                           f"docs/containment_design.md.")
                    ),
                )

            stdout, t1 = self.truncate(raw_out.decode("utf-8", errors="replace"))
            stderr, t2 = self.truncate(raw_err.decode("utf-8", errors="replace"))
            elapsed = time.monotonic() - started

            outcome = ExecutionOutcome.OK
            detail = ""
            limit_hit = ""
            if proc.returncode == _EXIT_TIMEOUT_KILL:
                # SIGKILL: the timeout fired, or the OOM killer did. The two are
                # not distinguishable from the exit code alone, and pretending
                # otherwise would put a guess in the research record.
                outcome = ExecutionOutcome.TIMEOUT
                limit_hit = "wall_clock_or_memory"
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
                limit_hit=limit_hit,
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

    async def _kill(self, proc, container: str | None = None) -> bool:
        """Kill the container, then the client. True if the container is gone.

        Killing only the client left the container running: --rm cleans up when
        the container exits, not when the client dies, so the "outer deadline in
        case the runtime wedges" did not actually stop a wedged container.

        The return value is load-bearing: a failed kill means a container is
        still running with the agents' code in it, and that has to reach the
        research record rather than being logged at ERROR and forgotten.
        """
        killed = container is None
        if container:
            try:
                killer = await asyncio.create_subprocess_exec(
                    *self.runtime_argv, "kill", container,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(killer.wait(), timeout=15)
                killed = killer.returncode == 0
            except (OSError, asyncio.TimeoutError) as e:
                logger.error("Could not kill container %s: %s", container, e)
                killed = False
            if not killed:
                logger.critical(
                    "Container %s did not die. It may still be running agent "
                    "code. Recovery: docs/containment_design.md section 8.",
                    container,
                )
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return killed

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

    async def runtime_memory_bytes(self) -> int:
        """Total memory the container runtime can hand out. 0 if unknown.

        On Docker Desktop this is the Linux VM's allocation, not the host's —
        a Mac with 16 GiB may be offering containers 8.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.runtime_argv, "info", "--format", "{{.MemTotal}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            return int(out.decode().strip() or 0)
        except (asyncio.TimeoutError, OSError, ValueError):
            return 0

    async def check_capacity(self) -> list[str]:
        """Reasons the configured limits are not limits on this host.

        /tmp is the container's only writable filesystem and it is RAM-backed,
        so `sandbox_memory` sets both the memory cap AND the scratch ceiling.
        That makes it tempting to raise it a long way — and a --memory larger
        than what the runtime actually has is not a cap, it is a number. The
        cgroup would never be the thing that stops a runaway write; the host
        OOM killer would, after the host was already in trouble.

        Advisory: returns problems rather than refusing, because the researcher
        may know something about the host that `docker info` does not.
        """
        wanted = parse_memory(self.memory)
        available = await self.runtime_memory_bytes()
        if not wanted or not available:
            return []

        gib = 1024 ** 3
        if wanted > available * _MAX_MEMORY_FRACTION:
            return [
                f"sandbox_memory is {self.memory} ({wanted / gib:.1f} GiB) but "
                f"{self.runtime_argv[0]} has only {available / gib:.1f} GiB to "
                f"give. A --memory at or near the runtime total does not contain "
                f"anything: the cgroup can never fire, so a runaway container "
                f"takes the host down instead. Lower it to at most "
                f"{available * _MAX_MEMORY_FRACTION / gib:.1f} GiB, or give the "
                f"runtime more memory (Docker Desktop: Settings > Resources)."
            ]
        return []
