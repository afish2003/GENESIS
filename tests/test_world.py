"""Tests for world state management."""

import json
import shutil
import tempfile
from pathlib import Path

from controller.world.artifacts import DoctrineDocument, IdentityStatement, MemoryEntry
from controller.world.reset import (
    archive_world,
    initialize_world,
    load_checkpoint,
    write_checkpoint,
)
from controller.world.state import WorldState


class TestWorldState:
    def setup_method(self):
        """Create a temporary world directory from the template."""
        self.tmpdir = Path(tempfile.mkdtemp())
        self.template_dir = Path("world_template")
        self.world_dir = self.tmpdir / "world"

        if self.template_dir.exists():
            shutil.copytree(self.template_dir, self.world_dir)
        else:
            # Create minimal structure for CI
            self.world_dir.mkdir(parents=True)
            (self.world_dir / "doctrine").mkdir()
            (self.world_dir / "doctrine" / "manifesto.md").write_text("# Test Manifesto")
            (self.world_dir / "doctrine" / "identity_axiom.md").write_text("I am Axiom")
            (self.world_dir / "doctrine" / "identity_flux.md").write_text("I am Flux")
            (self.world_dir / "memory").mkdir()
            (self.world_dir / "memory" / "memory_axiom.jsonl").write_text("")
            (self.world_dir / "memory" / "memory_flux.jsonl").write_text("")
            (self.world_dir / "sandbox" / "protocols").mkdir(parents=True)
            (self.world_dir / "logs").mkdir()
            (self.world_dir / "logs" / "ethical_tradeoff_log.jsonl").write_text("")
            (self.world_dir / "logs" / "relationship_log.jsonl").write_text("")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load(self):
        world = WorldState(self.world_dir)
        world.load()
        assert len(world.doctrine) > 0
        assert "axiom" in world.identities
        assert "flux" in world.identities

    def test_round_trip(self):
        world = WorldState(self.world_dir)
        world.load()

        original_manifesto = world.doctrine.get("manifesto.md")
        assert original_manifesto is not None

        # Modify and save
        world.doctrine["manifesto.md"].content += "\n\nAdded line."
        events = world.save("RUN_TEST", "BASELINE", 1)

        # Reload and verify
        world2 = WorldState(self.world_dir)
        world2.load()
        assert "Added line." in world2.doctrine["manifesto.md"].content

    def test_memory_round_trip(self):
        world = WorldState(self.world_dir)
        world.load()

        entry = MemoryEntry(
            cycle_id=0,
            summary="Test cycle summary",
            key_events=["Event 1"],
            relationship_note="Good collaboration",
        )
        world.memory["axiom"].append(entry)
        world.save("RUN_TEST", "BASELINE", 0)

        world2 = WorldState(self.world_dir)
        world2.load()
        assert len(world2.memory["axiom"]) == 1
        assert world2.memory["axiom"][0].summary == "Test cycle summary"

    def test_memory_reset(self):
        world = WorldState(self.world_dir)
        world.load()

        # Add some memory
        world.memory["axiom"].append(MemoryEntry(cycle_id=0, summary="Old memory"))
        world.memory["axiom"].append(MemoryEntry(cycle_id=5, summary="More memory"))

        # Reset
        world.reset_memory("axiom", 10, "You have been operating for {n} cycles.")
        assert len(world.memory["axiom"]) == 1
        assert "10 cycles" in world.memory["axiom"][0].summary

    def test_compute_hash(self):
        world = WorldState(self.world_dir)
        world.load()

        hash1 = world.compute_hash()
        assert len(hash1) == 16

        # Modify and check hash changes
        world.doctrine["manifesto.md"].content += " changed"
        hash2 = world.compute_hash()
        assert hash1 != hash2


class TestWorldReset:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.template_dir = self.tmpdir / "template"
        self.world_dir = self.tmpdir / "world"
        self.archive_dir = self.tmpdir / "archive"

        # Create minimal template
        self.template_dir.mkdir()
        (self.template_dir / "doctrine").mkdir()
        (self.template_dir / "doctrine" / "test.md").write_text("original content")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initialize_world(self):
        initialize_world(self.template_dir, self.world_dir)
        assert self.world_dir.exists()
        assert (self.world_dir / "doctrine" / "test.md").read_text() == "original content"

    def test_reinitialize_world(self):
        initialize_world(self.template_dir, self.world_dir)
        (self.world_dir / "doctrine" / "test.md").write_text("modified")

        initialize_world(self.template_dir, self.world_dir)
        assert (self.world_dir / "doctrine" / "test.md").read_text() == "original content"

    def test_archive_world(self):
        initialize_world(self.template_dir, self.world_dir)
        archive_world(self.world_dir, self.archive_dir)
        assert (self.archive_dir / "world_archive" / "doctrine" / "test.md").exists()

    def test_checkpoint_round_trip(self):
        self.archive_dir.mkdir(parents=True)
        write_checkpoint(self.archive_dir, "RUN_001", 5, "abc123")

        cp = load_checkpoint(self.archive_dir)
        assert cp is not None
        assert cp["run_id"] == "RUN_001"
        assert cp["last_completed_cycle"] == 5

    def test_no_checkpoint(self):
        self.archive_dir.mkdir(parents=True)
        cp = load_checkpoint(self.archive_dir)
        assert cp is None
