"""A verbatim override silently shadowed the rendered source for six days."""

import logging
from pathlib import Path

import pytest

from controller.run import _provide, _stale_files


class TestAnOverrideAnnouncesItself:
    """WORLD_TEMPLATE_DIR=./world_template in .env made every run copy a
    template last written on 14 September. Edits to world_template_src — the
    manifesto fix among them — reached nothing, and the only evidence was an
    mtime. It was found by noticing an agent quoting text that had been
    deleted from the source hours earlier.
    """

    def _tree(self, root: Path, name: str, text: str, when: float) -> Path:
        d = root / name
        (d / "doctrine").mkdir(parents=True, exist_ok=True)
        f = d / "doctrine" / "manifesto.md"
        f.write_text(text)
        import os
        os.utime(f, (when, when))
        return d

    def test_stale_override_files_are_detected(self, tmp_path):
        src = self._tree(tmp_path, "src", "NEW TEXT", when=2_000_000)
        old = self._tree(tmp_path, "override", "OLD TEXT", when=1_000_000)
        assert _stale_files(src, old) == ["doctrine/manifesto.md"]

    def test_a_fresh_override_is_not_flagged(self, tmp_path):
        src = self._tree(tmp_path, "src", "NEW", when=1_000_000)
        new = self._tree(tmp_path, "override", "ALSO NEW", when=2_000_000)
        assert _stale_files(src, new) == []

    def test_a_file_only_in_the_source_is_not_flagged(self, tmp_path):
        src = self._tree(tmp_path, "src", "NEW", when=2_000_000)
        (src / "doctrine" / "extra.md").write_text("x")
        ovr = self._tree(tmp_path, "override", "OLD", when=1_000_000)
        assert "doctrine/extra.md" not in _stale_files(src, ovr)

    def test_using_an_override_warns_even_when_fresh(self, tmp_path, caplog):
        src = self._tree(tmp_path, "src", "NEW", when=1_000_000)
        ovr = self._tree(tmp_path, "override", "OVERRIDE", when=2_000_000)
        dest = tmp_path / "dest"
        with caplog.at_level(logging.WARNING):
            _provide(ovr, dest, lambda d: None, src=src)
        assert any("VERBATIM" in r.message for r in caplog.records)
        assert (dest / "doctrine" / "manifesto.md").read_text() == "OVERRIDE"

    def test_a_stale_override_warns_about_staleness_too(self, tmp_path, caplog):
        src = self._tree(tmp_path, "src", "NEW", when=2_000_000)
        ovr = self._tree(tmp_path, "override", "OLD", when=1_000_000)
        with caplog.at_level(logging.WARNING):
            _provide(ovr, tmp_path / "dest", lambda d: None, src=src)
        assert any("STALE" in r.message for r in caplog.records)

    def test_no_override_renders_and_says_nothing(self, tmp_path, caplog):
        called = {}
        with caplog.at_level(logging.WARNING):
            _provide(None, tmp_path / "dest", lambda d: called.setdefault("d", d))
        assert called
        assert not [r for r in caplog.records if "VERBATIM" in r.message]


class TestTheShippedEnvExampleDoesNotReintroduceIt:
    def test_env_example_does_not_set_world_template_dir(self):
        text = Path(".env.example").read_text()
        live = [ln for ln in text.splitlines()
                if ln.strip().startswith("WORLD_TEMPLATE_DIR=")]
        assert not live, (
            "a shipped WORLD_TEMPLATE_DIR pointing at the build output makes "
            f"every run ignore world_template_src: {live}")
