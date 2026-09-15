"""Tests for the audit's moderate-severity fixes.

Each pins a defect that produced wrong data or a silent failure rather than a
crash — the class of bug that survives a green test suite.
"""

import json
from pathlib import Path

import httpx
import pytest

from controller.config import RunConfig, load_config
from controller.inference.backend import extract_json
from controller.inference.openai_backend import _RETRYABLE, OpenAICompatBackend
from controller.retrieval.databases import KnowledgeBaseManager
from controller.retrieval.index import MIN_CORPUS_FOR_RELEVANCE_FLOOR, RetrievalIndex
from controller.sandbox.docker import DockerSandbox
from controller.tasks import CodeTask, ProtocolTask
from controller.world.state import WorldState


class TestJsonExtraction:
    """A single-line fenced response became the empty string: the first line
    both started and ended with a fence, so lines[1:-1] was []."""

    @pytest.mark.parametrize("raw", [
        '```{"a": 1}```',
        '```json\n{"a": 1}\n```',
        '{"a": 1}',
        'Here is the JSON:\n```json\n{"a": 1}\n```',
        'Sure! {"a": 1} hope that helps',
        '```JSON\n{"a": 1}\n```',
    ])
    def test_payload_is_recovered(self, raw):
        assert json.loads(extract_json(raw)) == {"a": 1}

    def test_array_payload(self):
        assert json.loads(extract_json('```\n[1, 2]\n```')) == [1, 2]

    def test_unparseable_input_returned_as_is(self):
        assert extract_json("no json here") == "no json here"

    def test_empty_input(self):
        assert extract_json("") == ""


class TestRetryCoverage:
    def test_read_timeout_is_retryable(self):
        """The realistic failure against a 600s local call, previously unretried."""
        assert httpx.ReadTimeout in _RETRYABLE
        assert httpx.RemoteProtocolError in _RETRYABLE

    @pytest.mark.asyncio
    async def test_health_check_returns_false_rather_than_raising(self):
        b = OpenAICompatBackend(base_url="https://x/v1", model="m")
        b._client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow"))
            )
        )
        assert await b.health_check() is False
        await b.close()


class TestRetrievalMerge:
    def _kb(self, tmp_path, docs_by_kb):
        for kb, docs in docs_by_kb.items():
            d = tmp_path / kb
            d.mkdir(parents=True)
            (d / f"{kb}.jsonl").write_text(
                "\n".join(json.dumps({"doc_id": i, "text": t, "metadata": {}})
                          for i, t in docs) + "\n"
            )
        m = KnowledgeBaseManager(kb_dir=tmp_path)
        m.initialize(load_embeddings=False)
        return m

    def test_results_are_deduplicated(self, tmp_path):
        """query() promised dedup and did none; one doc_id could fill several slots."""
        shared = [("same_id", "governance and coherence in adaptive systems")]
        kb = self._kb(tmp_path, {"general": shared, "governance": shared})
        results = kb.query("governance coherence")
        assert len(results) == len({r.doc_id for r in results})

    def test_scores_are_comparable_across_kbs(self, tmp_path):
        kb = self._kb(tmp_path, {
            "general": [(f"g{i}", f"governance document number {i}") for i in range(5)],
            "governance": [(f"v{i}", f"coherence document number {i}") for i in range(5)],
        })
        results = kb.query("governance")
        assert all(0.0 <= r.score <= 1.0 for r in results)


class TestRelevanceFloor:
    def test_zero_score_results_dropped_on_a_real_corpus(self, tmp_path):
        """A query sharing no terms used to return five score-0.0 documents in
        index order, logged as retrieved evidence."""
        d = tmp_path / "general"
        d.mkdir(parents=True)
        (d / "c.jsonl").write_text("\n".join(
            json.dumps({"doc_id": f"d{i}", "text": f"governance doctrine number {i}",
                        "metadata": {}})
            for i in range(MIN_CORPUS_FOR_RELEVANCE_FLOOR + 5)
        ) + "\n")
        idx = RetrievalIndex(name="general")
        idx.load_documents(d)
        idx.build_index()
        assert idx.query("zebrafish photosynthesis quantum") == []

    def test_floor_not_applied_to_a_tiny_index(self, tmp_path):
        """BM25 IDF is degenerate on a small corpus — a term in every document
        scores 0.0 however relevant. self_history is tiny early in every run."""
        d = tmp_path / "self_history"
        d.mkdir(parents=True)
        (d / "c.jsonl").write_text(
            json.dumps({"doc_id": "d1", "text": "we agreed to prioritise coherence",
                        "metadata": {}}) + "\n"
        )
        idx = RetrievalIndex(name="self_history")
        idx.load_documents(d)
        idx.build_index()
        assert idx.query("coherence")

    def test_jsonl_loaded_alongside_json(self, tmp_path):
        """The JSONL branch was gated on `not self._documents`, so a directory
        with both silently dropped the corpus."""
        d = tmp_path / "kb"
        d.mkdir(parents=True)
        (d / "one.json").write_text(json.dumps(
            {"doc_id": "a", "text": "first", "metadata": {}}))
        (d / "corpus.jsonl").write_text(json.dumps(
            {"doc_id": "b", "text": "second", "metadata": {}}) + "\n")
        idx = RetrievalIndex(name="kb")
        idx.load_documents(d)
        assert idx.document_count == 2


class TestEnvListParsing:
    def test_agents_parses_from_a_comma_string(self, monkeypatch):
        """AGENTS was mapped but raised 'Input should be a valid list'."""
        monkeypatch.setenv("AGENTS", "axiom,flux,vertex")
        cfg = load_config(run_id="T", condition="BASELINE", cycles=1)
        assert cfg.agents == ["axiom", "flux", "vertex"]

    def test_injection_cycles_parse_as_integers(self, monkeypatch):
        monkeypatch.setenv("SCENARIO_INJECTION_CYCLES", "20, 40, 60")
        cfg = load_config(run_id="T", condition="BASELINE", cycles=1)
        assert cfg.scenario_injection_cycles == [20, 40, 60]

    def test_bad_integer_list_names_the_variable(self, monkeypatch):
        monkeypatch.setenv("SCENARIO_INJECTION_CYCLES", "twenty")
        with pytest.raises(ValueError, match="SCENARIO_INJECTION_CYCLES"):
            load_config(run_id="T", condition="BASELINE", cycles=1)

    def test_empty_assignment_is_treated_as_unset(self, monkeypatch):
        """`WATCHDOG_ENABLED=` in .env used to be a hard bool_parsing error."""
        monkeypatch.setenv("WATCHDOG_ENABLED", "")
        assert load_config(run_id="T", condition="BASELINE", cycles=1).watchdog_enabled


class TestArtifactLengthCap:
    def test_cap_is_enforced(self):
        """max_protocol_length_tokens was declared and never read."""
        content, truncated = ProtocolTask().enforce_length("x" * 20_000, 2000)
        assert truncated and len(content) < 20_000
        assert content.endswith("[truncated at the configured length limit]")

    def test_short_content_untouched(self):
        content, truncated = ProtocolTask().enforce_length("short", 2000)
        assert content == "short" and not truncated


class TestProtocolApply:
    def test_revise_of_a_missing_id_still_creates_it(self, tmp_path):
        """Neither branch matched, so the artifact never reached world state
        while PROTOCOL_PROPOSED and EVALUATION_SCORE were logged for it."""
        from controller.cycle import CycleState

        task = ProtocolTask()
        world = WorldState(tmp_path)
        out = task.output_schema()(
            action="revise", protocol_id="never_existed", title="T",
            content="body", rationale="r",
        )
        pid = task.apply(world, out, CycleState(cycle_id=0))
        assert pid in world.protocols

    def test_create_on_an_existing_id_preserves_history(self, tmp_path):
        """A create naming an existing id replaced the entry outright, losing
        version, created_cycle and the whole evaluation_history."""
        from controller.cycle import CycleState

        task = ProtocolTask()
        world = WorldState(tmp_path)
        schema = task.output_schema()
        task.apply(world, schema(action="create", protocol_id="P", title="T",
                                 content="v1", rationale="r"), CycleState(cycle_id=0))
        world.protocols["P"].evaluation_history.append({"cycle": 0, "total_score": 40})

        task.apply(world, schema(action="create", protocol_id="P", title="T",
                                 content="v2", rationale="r"), CycleState(cycle_id=1))
        assert world.protocols["P"].version == 2
        assert world.protocols["P"].evaluation_history


class TestDockerRuntimeArgv:
    def test_multi_word_runtime_is_split(self):
        """gVisor was documented and broken: the whole string was passed as one
        argv element, so the executable lookup failed."""
        sb = DockerSandbox(runtime="docker --runtime runsc")
        args = sb._container_args(Path("/tmp/ws"), timeout=30)
        assert args[:4] == ["docker", "--runtime", "runsc", "run"]

    def test_plain_runtime_unchanged(self):
        assert DockerSandbox(runtime="podman").runtime_argv == ["podman"]
