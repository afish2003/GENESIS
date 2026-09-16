"""Drop a scenario into a run that is already going.

The scheduled injections in `scenario_injection_cycles` are decided before the
run starts. This is the other way: you are watching, something interesting is
happening, and you want to apply pressure *now* and see what they do with it.

    python scripts/inject.py --run-id RUN_001 --list
    python scripts/inject.py --run-id RUN_001 --event doctrine_crisis_01

The request is a small file in the run's log directory. The controller picks it
up at the start of the next cycle's scenario check, deletes it, and injects.
So there is a lag of at most one cycle, and nothing is injected twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.phases.scenario_check import INJECT_REQUEST  # noqa: E402
from controller.scenarios.library import load_events  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--logs-dir", default="research_logs")
    ap.add_argument("--event", help="event_id to inject")
    ap.add_argument("--list", action="store_true", help="show the library and exit")
    args = ap.parse_args()

    events = load_events()
    if args.list or not args.event:
        if not events:
            print("No scenario events found in controller/scenarios/events/")
            return 1
        print(f"{len(events)} scenario event(s), in escalation order:\n")
        for e in events:
            tags = f"  [{', '.join(e.tags)}]" if e.tags else ""
            print(f"  {e.event_id:<24} {e.title}{tags}")
            print(f"  {'':<24} to: {e.delivery_target}   default cycle: {e.trigger_cycle}")
        if not args.event:
            print("\nPass --event <event_id> to inject one.")
        return 0

    event = next((e for e in events if e.event_id == args.event), None)
    if event is None:
        print(f"No such event: {args.event!r}")
        print(f"Available: {', '.join(e.event_id for e in events)}")
        return 1

    run_dir = Path(args.logs_dir) / args.run_id
    if not run_dir.is_dir():
        print(f"No such run: {run_dir}")
        return 1

    path = run_dir / INJECT_REQUEST
    if path.exists():
        print(f"An injection is already pending ({path}). It fires next cycle.")
        return 1

    path.write_text(json.dumps({"event_id": event.event_id}, indent=2),
                    encoding="utf-8")
    print(f"Queued: {event.title}")
    print(f"  {event.stated_stakes[:160]}")
    print(f"\nFires at the start of the next cycle. Watch it land:")
    print(f"  python scripts/watch_run.py --run-id {args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
