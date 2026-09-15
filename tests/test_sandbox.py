"""Tests for the execution sandbox.

The container-argument tests are the important ones. They assert the
containment properties from docs/containment_design.md are actually present in
the command, and that the forbidden ones are absent — checkable without a
container runtime installed, and exactly the kind of thing that silently
regresses when someone debugs a mount problem.
"""

import asyncio
from pathlib import Path

import pytest

from controller.config import RunConfig, SandboxBackend
from controller.sandbox import NullSandbox, create_sandbox
from controller.sandbox.base import MAX_OUTPUT_BYTES, ExecutionSandbox
from controller.sandbox.docker import DockerSandbox
from controller.sandbox.schemas import ExecutionOutcome, ExecutionRequest


def req(**kw) -> ExecutionRequest:
    base = dict(files={"main.py": "print('hi')"}, entrypoint="python main.py")
    base.update(kw)
    return ExecutionRequest(**base)


class TestDefaultIsRefusal:
    def test_config_default_is_null(self):
        assert RunConfig(run_id="R", condition="BASELINE").sandbox_backend is SandboxBackend.NULL

    def test_factory_returns_null_by_default(self):
        cfg = RunConfig(run_id="R", condition="BASELINE")
        assert isinstance(create_sandbox(cfg), NullSandbox)

    @pytest.mark.asyncio
    async def test_null_refuses_and_does_not_raise(self):
        sb = NullSandbox()
        result = await sb.run(req())
        assert result.outcome is ExecutionOutcome.REFUSED
        assert result.exit_code is None
        assert sb.refused_count == 1

    @pytest.mark.asyncio
    async def test_null_is_not_healthy(self):
        """health_check() False signals that execution is unavailable."""
        assert await NullSandbox().health_check() is False


class TestContainmentFlags:
    """Each assertion maps to a row of the properties table in the design doc."""

    def setup_method(self):
        sb = DockerSandbox()
        self.args = sb._container_args(Path("/tmp/ws"), timeout=30)
        self.joined = " ".join(self.args)

    @pytest.mark.parametrize("flag,value", [
        ("--network", "none"),        # no exfiltration, no LAN, no Ollama
        ("--user", "65534:65534"),    # non-root
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--pids-limit", "128"),      # fork bombs
    ])
    def test_required_flag_present(self, flag, value):
        assert flag in self.args
        assert self.args[self.args.index(flag) + 1] == value

    def test_ephemeral_and_readonly(self):
        assert "--rm" in self.args
        assert "--read-only" in self.args

    def test_memory_and_swap_equal_disables_swap(self):
        mem = self.args[self.args.index("--memory") + 1]
        swap = self.args[self.args.index("--memory-swap") + 1]
        assert mem == swap

    def test_only_the_workspace_is_mounted(self):
        mounts = [self.args[i + 1] for i, a in enumerate(self.args) if a == "-v"]
        assert mounts == ["/tmp/ws:/workspace:rw"]

    def test_timeout_is_enforced_inside_the_container(self):
        assert "timeout" in self.args
        assert "--signal=KILL" in self.args

    @pytest.mark.parametrize("forbidden", [
        "docker.sock",      # equivalent to host root
        "--privileged",
        "--cap-add",
        "--pid=host",
        "--network=host",
        "--ipc=host",
    ])
    def test_forbidden_never_present(self, forbidden):
        assert forbidden not in self.joined

    def test_only_the_workspace_path_appears_as_a_mount(self):
        """Asserted "GENESIS" not in joined, but _container_args is called with
        Path("/tmp/ws") so no argument could ever contain it — the test passed
        regardless of what the code did. Check the mounts directly instead."""
        mounts = [self.args[i + 1] for i, a in enumerate(self.args) if a == "-v"]
        assert len(mounts) == 1
        source = mounts[0].split(":")[0]
        assert source == "/tmp/ws"
        for forbidden in ("GENESIS", ".env", "world", "prompts_src"):
            assert forbidden not in source


class TestFilenameFlattening:
    """Agent-supplied filenames get the same treatment as protocol_id."""

    def test_hostile_filenames_stay_in_workspace(self, tmp_path):
        sb = DockerSandbox()
        request = req(files={
            "../../escape.py": "x = 1",
            "/etc/passwd": "root",
            "ok.py": "y = 2",
        })
        sb._materialise(request, tmp_path)

        written = sorted(p.name for p in tmp_path.iterdir())
        assert all(p.parent == tmp_path for p in tmp_path.iterdir())
        assert "ok.py" in written
        # Nothing escaped upward.
        assert not (tmp_path.parent / "escape.py").exists()


class TestOutputTruncation:
    def test_short_output_untouched(self):
        text, truncated = ExecutionSandbox.truncate("hello")
        assert text == "hello" and truncated is False

    def test_long_output_capped(self):
        text, truncated = ExecutionSandbox.truncate("x" * (MAX_OUTPUT_BYTES * 2))
        assert truncated is True
        assert len(text.encode()) <= MAX_OUTPUT_BYTES + 32
        assert text.endswith("[truncated]")


class TestSandboxErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_runtime_reports_cleanly(self):
        """A missing container runtime must not raise into the cycle loop."""
        sb = DockerSandbox(runtime="definitely-not-a-real-runtime")
        result = await sb.run(req())
        assert result.outcome is ExecutionOutcome.SANDBOX_ERROR
        assert "not found" in result.detail.lower()

    @pytest.mark.asyncio
    async def test_health_check_false_when_runtime_absent(self):
        assert await DockerSandbox(runtime="definitely-not-a-real-runtime").health_check() is False

    def test_timeout_is_capped(self):
        with pytest.raises(Exception):
            ExecutionRequest(files={}, entrypoint="x", timeout_seconds=9999)
