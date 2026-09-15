"""Mock inference backend — exercises the full cycle loop with no model.

`complete_structured()` in the base class appends the target JSON schema to the
final message before calling `complete()`. This backend reads that schema back
out and synthesises a conforming instance, so a single generic fake satisfies
all fourteen phases without hand-written canned responses.

Use it to validate control flow, schema round-trips, logging, world-state
diffing, checkpoint/resume, and memory reset — everything except whether a real
model produces good text. Not a substitute for a real run.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from controller.inference.backend import InferenceBackend, InferenceResult, Message

logger = logging.getLogger(__name__)

_LOREM = (
    "This is mock output produced without a language model. It exists to "
    "exercise the cycle loop end to end."
)


#: Values for fields whose name implies a constrained domain the JSON schema
#: does not express. Without these the mock emits lorem for `target_document`,
#: every doctrine revision fails to resolve, and an integration test can never
#: exercise the path where a revision actually lands.
DEFAULT_FIELD_HINTS: dict[str, str] = {
    "target_document": "doctrine.md",
    "protocol_id": "mock_protocol",
    "artifact_id": "mock_artifact",
}


class MockBackend(InferenceBackend):
    """Deterministic fake backend. Returns schema-valid JSON, or prose."""

    def __init__(
        self,
        model: str = "mock",
        seed: int = 0,
        field_hints: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.call_count = 0
        self._seed = seed
        self.field_hints = {**DEFAULT_FIELD_HINTS, **(field_hints or {})}

    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> InferenceResult:
        self.call_count += 1
        content = messages[-1].content if messages else ""

        schema = _extract_schema(content)
        if schema is not None:
            payload = _synthesize(schema, schema, hints=self.field_hints)
            text = json.dumps(payload)
        else:
            # Free-form phase (e.g. a discussion turn)
            text = f"{_LOREM} (call {self.call_count})"

        return InferenceResult(
            content=text,
            model=self.model,
            total_duration_ms=0,
            prompt_tokens=sum(len(m.content) // 4 for m in messages),
            completion_tokens=len(text) // 4,
        )

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def _extract_schema(content: str) -> dict[str, Any] | None:
    """Recover the JSON schema appended by complete_structured(), if present."""
    marker = "Schema: "
    idx = content.rfind(marker)
    if idx == -1:
        return None
    raw = content[idx + len(marker):].strip()
    # The schema is a Python dict repr or JSON; try JSON first.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        import ast

        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, SyntaxError):
        logger.debug("MockBackend could not parse appended schema")
        return None


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a local JSON-schema $ref like '#/$defs/EvaluationScores'."""
    if not ref.startswith("#/"):
        logger.debug("MockBackend cannot resolve external $ref %r", ref)
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            logger.debug("MockBackend failed to resolve $ref %r", ref)
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def _synthesize(
    node: dict[str, Any],
    root: dict[str, Any],
    depth: int = 0,
    hints: dict[str, str] | None = None,
) -> Any:
    """Build a minimal value satisfying a JSON-schema node."""
    if depth > 8:
        return None

    if "$ref" in node:
        resolved = _resolve_ref(node["$ref"], root)
        return _synthesize(resolved, root, depth + 1, hints) if resolved else None

    # Choose the first branch of a union that isn't null.
    for key in ("anyOf", "oneOf", "allOf"):
        if key in node:
            for option in node[key]:
                if option.get("type") != "null":
                    return _synthesize(option, root, depth + 1, hints)
            return None

    if "enum" in node and node["enum"]:
        return node["enum"][0]

    if "const" in node:
        return node["const"]

    node_type = node.get("type")

    if node_type == "object" or "properties" in node:
        props: dict[str, Any] = node.get("properties", {})
        required = set(node.get("required", props.keys()))
        # Hinted fields are emitted even when optional: a schema default of ""
        # would otherwise silently drop the value the caller asked for.
        return {
            name: (hints[name] if hints and name in hints
                   else _synthesize(spec, root, depth + 1, hints))
            for name, spec in props.items()
            if name in required or (hints and name in hints)
        }

    if node_type == "array":
        items = node.get("items")
        min_items = node.get("minItems", 1)
        if not items:
            return []
        element = _synthesize(items, root, depth + 1, hints)
        return [element] * max(min_items, 1)

    if node_type == "integer":
        return _satisfy_numeric(node, integer=True)
    if node_type == "number":
        return _satisfy_numeric(node, integer=False)
    if node_type == "boolean":
        return True
    if node_type == "null":
        return None

    # Default: string. Honour a pattern if one is declared.
    return _satisfy_string(node)


def _satisfy_numeric(node: dict[str, Any], integer: bool) -> Any:
    low = node.get("minimum", node.get("exclusiveMinimum", 0))
    high = node.get("maximum", node.get("exclusiveMaximum"))
    value = low if low is not None else 0
    if node.get("exclusiveMinimum") is not None:
        value = node["exclusiveMinimum"] + 1
    if high is not None and value > high:
        value = high
    return int(value) if integer else float(value)


def _satisfy_string(node: dict[str, Any]) -> str:
    pattern = node.get("pattern")
    if pattern:
        # Handle the common "^(a|b|c)$" enumeration idiom used in this codebase.
        match = re.fullmatch(r"\^\(([^)]+)\)\$", pattern)
        if match:
            return match.group(1).split("|")[0]
        literal = re.fullmatch(r"\^([A-Za-z0-9_\- ]+)\$", pattern)
        if literal:
            return literal.group(1)
        logger.debug("MockBackend cannot satisfy pattern %r; emitting empty string", pattern)
        return ""
    min_len = node.get("minLength", 0)
    text = _LOREM
    return text if len(text) >= min_len else text + "x" * (min_len - len(text))
