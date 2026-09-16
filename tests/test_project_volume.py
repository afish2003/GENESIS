"""Tests for the persistent, quota-capped project volume.

The volume exists so agents can build a codebase across cycles rather than one
throwaway module per cycle. It is the only writable, long-lived thing they get,
so most of these tests are about what the controller REFUSES: a bind mount has
no size limit, and Docker cannot give it one (`--storage-opt size=` is accepted
and ignored on Docker Desktop's overlayfs — measured at 1600 MiB written under
`size=1G`). The cap has to come from the filesystem being genuinely that size,
and the only way to be sure of that is to check the filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from controller.config import RunConfig
from controller.sandbox.docker import DockerSandbox
from controller.sandbox.project import (
    ProjectVolumeError,
    parse_size,
    resolve_project_volume,
)

REPO = Path(__file__).parent.parent


def cfg(tmp_path, **kw) -> RunConfig:
    base = dict(
        run_id="P", condition="BASELINE", sandbox_backend="docker",
        execution_enabled=True,
        world_dir=tmp_path / "world",
        research_logs_dir=tmp_path / "logs",
    )
    base.update(kw)
    return RunConfig(**base)


def pretend_filesystem_size(monkeypatch, total_bytes: int):
    """Make statvfs report a capped filesystem, so the accept path is testable.

    Creating a real one needs hdiutil or a loop mount; that is
    tests/test_containment_live.py's job.
    """
    real = os.statvfs

    def fake(path):
        st = real(path)
        return os.statvfs_result((
            st.f_bsize, st.f_frsize, total_bytes // st.f_frsize,
            total_bytes // st.f_frsize, total_bytes // st.f_frsize,
            st.f_files, st.f_ffree, st.f_favail, st.f_flag, st.f_namemax,
        ))

    monkeypatch.setattr(os, "statvfs", fake)


class TestParseSize:
    @pytest.mark.parametrize("text,expected", [
        ("32g", 32 * 1024 ** 3), ("512m", 512 * 1024 ** 2),
        ("1.5t", int(1.5 * 1024 ** 4)), ("2048", 2048), ("", 0), ("nonsense", 0),
    ])
    def test_parses(self, text, expected):
        assert parse_size(text) == expected


class TestNoVolumeByDefault:
    def test_unset_means_none(self, tmp_path):
        assert resolve_project_volume(cfg(tmp_path)) is None

    def test_config_default_is_none(self):
        assert RunConfig(run_id="P", condition="BASELINE").sandbox_project_dir is None

    def test_sandbox_mounts_only_the_workspace_without_one(self):
        args = DockerSandbox()._container_args(Path("/tmp/ws"), timeout=30)
        mounts = [args[i + 1] for i, a in enumerate(args) if a == "-v"]
        assert mounts == ["/tmp/ws:/workspace:ro"]
        assert args[args.index("-w") + 1] == "/workspace"


class TestRefusesUncappedStorage:
    """The whole point. A bind mount Docker cannot cap is the host's disk."""

    def test_refuses_an_ordinary_directory(self, tmp_path):
        """tmp_path is on the root filesystem — hundreds of GiB, not 32."""
        project = tmp_path / "project"
        project.mkdir()
        with pytest.raises(ProjectVolumeError, match="not a capped volume"):
            resolve_project_volume(cfg(tmp_path, sandbox_project_dir=project))

    def test_the_refusal_says_how_to_fix_it(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        with pytest.raises(ProjectVolumeError) as e:
            resolve_project_volume(cfg(tmp_path, sandbox_project_dir=project))
        assert "setup_project_volume.py" in str(e.value)

    def test_missing_directory_is_refused_not_created(self, tmp_path):
        """Creating it would produce an uncapped directory, silently."""
        missing = tmp_path / "nope"
        with pytest.raises(ProjectVolumeError, match="does not exist"):
            resolve_project_volume(cfg(tmp_path, sandbox_project_dir=missing))
        assert not missing.exists()

    def test_accepts_a_capped_filesystem(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        pretend_filesystem_size(monkeypatch, 32 * 1024 ** 3)
        volume = resolve_project_volume(
            cfg(tmp_path, sandbox_project_dir=project, sandbox_project_size="32g"))
        assert volume is not None and volume.path == project.resolve()

    def test_a_smaller_volume_than_declared_is_allowed(self, tmp_path, monkeypatch):
        """Still capped, just smaller. A warning, not a refusal."""
        project = tmp_path / "project"
        project.mkdir()
        pretend_filesystem_size(monkeypatch, 2 * 1024 ** 3)
        assert resolve_project_volume(
            cfg(tmp_path, sandbox_project_dir=project, sandbox_project_size="32g"))


class TestRefusesDangerousLocations:
    """A writable mount inside any of these ends the run's evidentiary value."""

    def _refused(self, tmp_path, monkeypatch, target: Path):
        target.mkdir(parents=True, exist_ok=True)
        pretend_filesystem_size(monkeypatch, 32 * 1024 ** 3)
        with pytest.raises(ProjectVolumeError) as e:
            resolve_project_volume(
                cfg(tmp_path, sandbox_project_dir=target, sandbox_project_size="32g"))
        return str(e.value)

    def test_refuses_the_world_directory(self, tmp_path, monkeypatch):
        """Doctrine and memory are controller-mediated; direct write access
        would let agents forge their own history."""
        msg = self._refused(tmp_path, monkeypatch, tmp_path / "world" / "sub")
        assert "world directory" in msg

    def test_refuses_the_research_logs(self, tmp_path, monkeypatch):
        msg = self._refused(tmp_path, monkeypatch, tmp_path / "logs" / "sub")
        assert "append-only" in msg

    def test_refuses_the_repo(self, tmp_path, monkeypatch):
        """Agents reading the controller that measures them is a confound even
        before it is a safety problem."""
        pretend_filesystem_size(monkeypatch, 32 * 1024 ** 3)
        with pytest.raises(ProjectVolumeError, match="GENESIS source"):
            resolve_project_volume(cfg(
                tmp_path, sandbox_project_dir=REPO / "controller",
                sandbox_project_size="32g"))


class TestMountedCorrectly:
    def setup_method(self):
        self.args = DockerSandbox(project_dir=Path("/mnt/proj"))._container_args(
            Path("/tmp/ws"), timeout=30)

    def test_project_is_writable_and_workspace_is_not(self):
        mounts = [self.args[i + 1] for i, a in enumerate(self.args) if a == "-v"]
        assert mounts == ["/mnt/proj:/project:rw", "/tmp/ws:/workspace:ro"]

    def test_code_runs_inside_the_project(self):
        """A program that builds a codebase has to run in the codebase."""
        assert self.args[self.args.index("-w") + 1] == "/project"

    def test_still_nothing_else_is_mounted(self):
        mounts = [self.args[i + 1] for i, a in enumerate(self.args) if a == "-v"]
        assert len(mounts) == 2
        assert "docker.sock" not in " ".join(self.args)

    def test_every_other_containment_flag_survives(self):
        for flag in ("--network", "--read-only", "--cap-drop", "--rm",
                     "--security-opt", "--pids-limit"):
            assert flag in self.args


class TestAgentsAreToldAboutIt:
    def test_prompt_mentions_persistence_when_there_is_a_volume(self, tmp_path):
        from controller.tasks import CodeTask

        project = tmp_path / "project"
        project.mkdir()
        (project / "app.py").write_text("x = 1")
        (project / "lib").mkdir()

        class W:
            protocols: dict = {}

        prompt = CodeTask().design_prompt(
            cfg(tmp_path, sandbox_project_dir=project), W(), None)
        assert "SURVIVES between cycles" in prompt
        assert "app.py" in prompt and "lib/" in prompt

    def test_no_project_note_without_a_volume(self, tmp_path):
        from controller.tasks import CodeTask

        class W:
            protocols: dict = {}

        prompt = CodeTask().design_prompt(cfg(tmp_path), W(), None)
        assert "/project" not in prompt


class TestPathExpansion:
    """`~` must expand once, at config load, not per consumer.

    It was resolved in two places and expanded in one. resolve_project_volume
    called .expanduser(), so the volume mounted and the container really did get
    /project — while CodeTask._project_tree did a bare Path(root).is_dir() on
    the literal string "~/genesis_project", got False, and returned an empty
    tree. The agents were never told the directory existed.

    Every visible signal said healthy: volume verified, mount present, sandbox
    configured. The only symptom was an empty project directory, which is
    indistinguishable from agents deciding not to use it — and that is what I
    concluded from it, twice.
    """

    def test_a_tilde_path_is_expanded(self):
        config = RunConfig(run_id="P", condition="BASELINE",
                           sandbox_project_dir="~/genesis_project")
        assert str(config.sandbox_project_dir).startswith(str(Path.home()))
        assert "~" not in str(config.sandbox_project_dir)

    def test_an_absolute_path_is_left_alone(self, tmp_path):
        config = RunConfig(run_id="P", condition="BASELINE",
                           sandbox_project_dir=tmp_path)
        assert config.sandbox_project_dir == tmp_path

    def test_none_stays_none(self):
        config = RunConfig(run_id="P", condition="BASELINE")
        assert config.sandbox_project_dir is None

    def test_the_design_prompt_sees_the_same_directory_the_sandbox_mounts(
            self, tmp_path, monkeypatch):
        """The two consumers must agree. This is the actual bug."""
        from controller.tasks import CodeTask

        project = tmp_path / "proj"
        project.mkdir()
        (project / "built_last_cycle.py").write_text("x = 1")

        # expanduser() reads $HOME, not Path.home(), so patch the environment.
        monkeypatch.setenv("HOME", str(tmp_path))
        config = RunConfig(
            run_id="P", condition="BASELINE", execution_enabled=True,
            sandbox_backend="docker", sandbox_project_dir="~/proj",
            world_dir=tmp_path / "w", research_logs_dir=tmp_path / "l",
        )
        pretend_filesystem_size(monkeypatch, 32 * 1024 ** 3)

        volume = resolve_project_volume(config)
        tree = CodeTask._project_tree(config)
        assert volume is not None
        assert "built_last_cycle.py" in tree
        assert volume.path == project.resolve()
