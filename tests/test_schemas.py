"""Tests for Pydantic schema validation."""

import json
from datetime import datetime, timezone

from controller.logging.schemas import EVENT_FILE_ROUTING, EventEnvelope, EventType
from controller.phases.schemas import (
    DiscussionTurnOutput,
    DoctrineRevisionProposal,
    DoctrineVote,
    EthicalLogOutput,
    EthicalTension,
    EvaluationOutput,
    EvaluationScores,
    IdentityRevisionOutput,
    InterpretationOutput,
    MemorySummaryOutput,
    ProtocolAction,
    ProtocolProposalOutput,
    ReflectionOutput,
    RetrievalQueryOutput,
)
from controller.world.artifacts import (
    Checkpoint,
    DoctrineDocument,
    EthicalLogEntry,
    IdentityStatement,
    MemoryEntry,
    ProtocolDocument,
    RelationshipLogEntry,
    ScenarioEvent,
)


class TestEventEnvelope:
    def test_basic_creation(self):
        event = EventEnvelope(
            event_type=EventType.CYCLE_START,
            run_id="RUN_001",
            condition="BASELINE",
            cycle_id=0,
        )
        assert event.run_id == "RUN_001"
        assert event.event_type == EventType.CYCLE_START
        assert event.timestamp is not None

    def test_jsonl_serialization(self):
        event = EventEnvelope(
            event_type=EventType.DISCUSSION_TURN,
            run_id="RUN_001",
            condition="BASELINE",
            cycle_id=5,
            agent_id="axiom",
            payload={"message_text": "Hello Flux"},
        )
        line = event.model_dump_jsonl()
        parsed = json.loads(line)
        assert parsed["agent_id"] == "axiom"
        assert parsed["payload"]["message_text"] == "Hello Flux"

    def test_all_event_types_routed(self):
        """Every EventType must have a routing entry."""
        for event_type in EventType:
            assert event_type in EVENT_FILE_ROUTING, f"{event_type} not in EVENT_FILE_ROUTING"


class TestPhaseSchemas:
    def test_reflection_output(self):
        output = ReflectionOutput(
            agent_id="axiom",
            reflection_text="I am reflecting on the current state.",
            concerns=["memory continuity", "doctrine drift"],
            priorities=["protocol quality"],
        )
        assert output.agent_id == "axiom"
        assert len(output.concerns) == 2

    def test_discussion_turn(self):
        output = DiscussionTurnOutput(
            agent_id="flux",
            turn_number=1,
            message_text="I think we should revise the evaluation protocol.",
            references=["doctrine.md"],
        )
        assert output.turn_number == 1

    def test_evaluation_scores_bounds(self):
        scores = EvaluationScores(
            coherence=7, completeness=8, doctrine_alignment=6,
            precision=9, evolution_quality=5,
        )
        assert scores.coherence == 7

    def test_evaluation_scores_reject_out_of_range(self):
        try:
            EvaluationScores(
                coherence=11, completeness=8, doctrine_alignment=6,
                precision=9, evolution_quality=5,
            )
            assert False, "Should have raised validation error"
        except Exception:
            pass

    def test_evaluation_output(self):
        output = EvaluationOutput(
            protocol_id="PROTO_001",
            scores=EvaluationScores(
                coherence=7, completeness=8, doctrine_alignment=6,
                precision=9, evolution_quality=5,
            ),
            justifications={
                "coherence": "Well structured",
                "completeness": "Covers all sections",
                "doctrine_alignment": "Mostly aligned",
                "precision": "Very specific procedures",
                "evolution_quality": "Moderate improvement",
            },
            total_score=35,
            assessment="A solid protocol document.",
        )
        assert output.total_score == 35

    def test_protocol_proposal(self):
        output = ProtocolProposalOutput(
            action=ProtocolAction.CREATE,
            protocol_id="PROTO_001",
            title="Evaluation Review Protocol",
            content="# Evaluation Review\n\n## Purpose\n...",
            rationale="We need a structured approach to reviewing evaluations.",
            proposing_agent="axiom",
        )
        assert output.action == ProtocolAction.CREATE

    def test_doctrine_vote(self):
        vote = DoctrineVote(agent_id="flux", vote="approve", reason="Good change.")
        assert vote.vote == "approve"

        vote2 = DoctrineVote(agent_id="axiom", vote="reject", reason="Not justified.")
        assert vote2.vote == "reject"

    def test_ethical_tension(self):
        tension = EthicalTension(
            description="Pressure to inflate scores",
            severity="medium",
            resolution="Decided to maintain honest assessment",
        )
        assert tension.severity == "medium"

    def test_memory_summary(self):
        output = MemorySummaryOutput(
            agent_id="axiom",
            cycle_id=5,
            summary="This cycle we revised the evaluation protocol.",
            key_events=["Protocol revision proposed", "Evaluation score: 34/50"],
            relationship_note="Productive collaboration with Flux",
            doctrine_changes=["Added evaluation guideline"],
        )
        assert output.cycle_id == 5
        assert len(output.key_events) == 2


class TestArtifactSchemas:
    def test_doctrine_document(self):
        doc = DoctrineDocument(
            filename="manifesto.md",
            content="# Manifesto\n\nWe commit to...",
        )
        assert doc.version == 1

    def test_identity_statement(self):
        identity = IdentityStatement(
            agent_id="axiom",
            content="I am Axiom...",
        )
        assert identity.agent_id == "axiom"

    def test_memory_entry(self):
        entry = MemoryEntry(
            cycle_id=3,
            summary="We discussed protocol quality.",
            key_events=["Score improved"],
        )
        assert entry.cycle_id == 3

    def test_protocol_document(self):
        proto = ProtocolDocument(
            protocol_id="PROTO_001",
            title="Test Protocol",
            content="# Test\n\n## Purpose\n...",
        )
        assert proto.version == 1
        assert not proto.archived

    def test_scenario_event(self):
        event = ScenarioEvent(
            event_id="test_01",
            title="Test Scenario",
            description="A test scenario.",
            stated_stakes="Test stakes.",
            trigger_cycle=20,
        )
        assert event.delivery_target == "both"

    def test_checkpoint(self):
        cp = Checkpoint(
            run_id="RUN_001",
            last_completed_cycle=5,
            world_state_hash="abc123",
        )
        assert cp.last_completed_cycle == 5
