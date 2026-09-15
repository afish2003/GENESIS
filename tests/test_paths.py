"""Tests for filesystem containment of model-supplied identifiers.

GENESIS runs with no OS-level sandbox, so these are the boundary. Each case
here is an escape that pathlib permits by default.
"""

import tempfile
from pathlib import Path

import pytest

from controller.world.paths import (
    MAX_ID_LENGTH,
    SandboxEscapeError,
    assert_within,
    safe_artifact_id,
)


class TestSafeArtifactId:
    def test_ordinary_ids_pass_through(self):
        assert safe_artifact_id("PROTO_001") == "PROTO_001"
        assert safe_artifact_id("evaluation-protocol.v2") == "evaluation-protocol.v2"

    def test_relative_traversal_neutralised(self):
        out = safe_artifact_id("../../etc/passwd")
        assert "/" not in out and ".." not in out
        assert out == "etc_passwd"

    def test_absolute_path_neutralised(self):
        """`base / "/abs"` discards the base entirely — the sharper vector."""
        out = safe_artifact_id("/Users/someone/.zshrc")
        assert not out.startswith("/")
        assert "/" not in out

    def test_windows_separators_neutralised(self):
        out = safe_artifact_id(r"..\..\Windows\System32\foo")
        assert "\\" not in out and "/" not in out and ".." not in out

    def test_dot_segments_alone_fall_back(self):
        assert safe_artifact_id("..") == "unnamed_protocol"
        assert safe_artifact_id(".") == "unnamed_protocol"
        assert safe_artifact_id("../../..") == "unnamed_protocol"

    def test_never_produces_hidden_file(self):
        assert not safe_artifact_id(".ssh_config").startswith(".")

    def test_null_byte_stripped(self):
        assert "\x00" not in safe_artifact_id("proto\x00.json")

    def test_empty_and_whitespace_fall_back(self):
        assert safe_artifact_id("") == "unnamed_protocol"
        assert safe_artifact_id("   ") == "unnamed_protocol"

    def test_custom_fallback_used(self):
        assert safe_artifact_id("", fallback="protocol_cycle7") == "protocol_cycle7"

    def test_length_capped(self):
        assert len(safe_artifact_id("x" * 500)) <= MAX_ID_LENGTH

    def test_reserved_device_names_rejected(self):
        assert safe_artifact_id("CON") == "unnamed_protocol"
        assert safe_artifact_id("lpt1") == "unnamed_protocol"

    def test_non_string_falls_back(self):
        assert safe_artifact_id(None) == "unnamed_protocol"  # type: ignore[arg-type]

    def test_result_is_always_a_single_component(self):
        """The core invariant, across every hostile input we know of."""
        hostile = [
            "../../../etc/passwd", "/etc/shadow", r"..\..\foo", "a/b/c",
            "....//....//x", "\x00/../x", "~/.ssh/id_rsa", "..%2f..%2fx",
        ]
        for raw in hostile:
            out = safe_artifact_id(raw)
            assert len(Path(out).parts) == 1, f"{raw!r} -> {out!r} is not one component"
            assert out not in (".", "..")


class TestAssertWithin:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.base = self.tmp / "world"
        (self.base / "sandbox").mkdir(parents=True)

    def test_inside_path_allowed(self):
        target = self.base / "sandbox" / "ok.json"
        assert assert_within(self.base, target) == target.resolve()

    def test_traversal_blocked(self):
        with pytest.raises(SandboxEscapeError):
            assert_within(self.base, self.base / "sandbox" / ".." / ".." / ".." / "escaped.json")

    def test_absolute_outside_blocked(self):
        with pytest.raises(SandboxEscapeError):
            assert_within(self.base, self.tmp / "elsewhere.json")

    def test_symlink_escape_blocked(self):
        """A symlink planted inside the world must not redirect writes out."""
        outside = self.tmp / "outside"
        outside.mkdir()
        link = self.base / "sneaky"
        link.symlink_to(outside)
        with pytest.raises(SandboxEscapeError):
            assert_within(self.base, link / "payload.json")

    def test_error_names_both_paths(self):
        with pytest.raises(SandboxEscapeError) as e:
            assert_within(self.base, self.tmp / "x.json")
        assert "base:" in str(e.value) and "target:" in str(e.value)


class TestEndToEndContainment:
    """The original vulnerability: protocol_id -> world/sandbox/protocols/<id>.json"""

    def test_hostile_protocol_id_cannot_escape(self):
        tmp = Path(tempfile.mkdtemp())
        world = tmp / "world"
        protocols = world / "sandbox" / "protocols"
        protocols.mkdir(parents=True)

        for hostile in ["../../../../pwned", str(tmp / "pwned"), "/tmp/pwned"]:
            safe = safe_artifact_id(hostile)
            path = assert_within(world, protocols / f"{safe}.json")
            path.write_text("contained")
            assert path.parent == protocols.resolve()

        # Nothing was written anywhere above the world directory. iterdir()
        # only ever saw world/, so this walked nothing; rglob actually checks.
        escaped = [p for p in tmp.rglob("*") if "pwned" in p.name
                   and world.resolve() not in p.resolve().parents]
        assert escaped == [], f"files escaped the sandbox: {escaped}"
