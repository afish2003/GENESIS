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


class TestEachRunGetsItsOwnWorld:
    """A shared ./world is destroyed by the next run.

    initialize_world does shutil.rmtree before copying the template, so run
    N+1 deleted run N's final doctrine and identities. run_battery.py set
    WORLD_DIR per run to avoid it; a plain `python -m controller.main` did
    not, and silently lost the world it had just built.
    """

    def test_the_default_is_per_run(self):
        from controller.config import RunConfig
        a = RunConfig(run_id="RUN_A", condition="BASELINE").world_dir
        b = RunConfig(run_id="RUN_B", condition="BASELINE").world_dir
        assert a != b
        assert "RUN_A" in str(a) and "RUN_B" in str(b)

    def test_an_explicit_directory_still_wins(self):
        """Sharing one on purpose stays possible — it just is not the default."""
        from controller.config import RunConfig
        cfg = RunConfig(run_id="R", condition="BASELINE", world_dir="./shared")
        assert str(cfg.world_dir) == "shared"

    def test_resume_resolves_to_the_same_place(self):
        """Derived from run_id, so a resumed run finds the world it left."""
        from controller.config import RunConfig
        first = RunConfig(run_id="SAME", condition="BASELINE").world_dir
        again = RunConfig(run_id="SAME", condition="BASELINE").world_dir
        assert first == again

    def test_neither_env_file_pins_a_shared_world(self):
        from pathlib import Path
        for name in (".env.example", ".env"):
            p = Path(name)
            if not p.exists():
                continue
            live = [ln for ln in p.read_text().splitlines()
                    if ln.strip().startswith("WORLD_DIR=")]
            assert not live, f"{name} pins a shared world dir: {live}"
