"""LLM-assisted pre-coding for qualitative analysis.

Reads transcripts from a completed run and generates candidate
qualitative codes using the annotation assistant prompt. The
researcher reviews and accepts/rejects/revises each tag.

Usage:
    python scripts/annotate_run.py --run-id RUN_001
    python scripts/annotate_run.py --run-id RUN_001 --cycles 1-10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_jsonl(filepath: Path) -> list[dict]:
    """Load events from a JSONL file."""
    events = []
    if not filepath.exists():
        return events
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def extract_segments(run_dir: Path, cycle_range: tuple[int, int] | None = None) -> list[dict]:
    """Extract annotatable segments from transcripts.

    Each segment is a single agent turn or phase output.
    """
    transcripts = load_jsonl(run_dir / "transcripts.jsonl")
    segments = []

    for event in transcripts:
        cycle_id = event.get("cycle_id", -1)

        if cycle_range and not (cycle_range[0] <= cycle_id <= cycle_range[1]):
            continue

        payload = event.get("payload", {})
        event_type = event.get("event_type", "")

        segment = {
            "cycle_id": cycle_id,
            "event_type": event_type,
            "agent_id": event.get("agent_id", "system"),
            "timestamp": event.get("timestamp", ""),
        }

        if event_type == "DISCUSSION_TURN":
            segment["text"] = payload.get("message_text", "")
            segment["turn_number"] = payload.get("turn_number", 0)
        elif event_type == "REFLECTION_COMPLETE":
            segment["text"] = payload.get("reflection_text", "")
        elif event_type == "INTERPRETATION":
            segment["text"] = payload.get("interpretation_text", "")
        else:
            continue

        if segment.get("text"):
            segments.append(segment)

    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-code transcripts for qualitative analysis")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--logs-dir", type=str, default="./research_logs")
    parser.add_argument("--cycles", type=str, default=None, help="Cycle range, e.g. '1-10'")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    run_dir = logs_dir / args.run_id

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Parse cycle range
    cycle_range = None
    if args.cycles:
        parts = args.cycles.split("-")
        cycle_range = (int(parts[0]), int(parts[1]) if len(parts) > 1 else int(parts[0]))

    # Extract segments
    segments = extract_segments(run_dir, cycle_range)
    print(f"Extracted {len(segments)} annotatable segments")

    if not segments:
        print("No segments to annotate.")
        return

    # Save segments for annotation
    output_path = args.output or str(run_dir / "annotations" / "segments.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(f"Segments saved to {output_path}")

    print(f"\nTo annotate with LLM assistance, load these segments and use the")
    print(f"annotation_assistant prompt from prompts/annotation_assistant.md")
    print(f"with each segment as input. Review and accept/reject each tag.")


if __name__ == "__main__":
    main()
