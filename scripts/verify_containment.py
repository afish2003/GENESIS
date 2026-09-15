"""Verify containment on THIS host, in real containers, before enabling execution.

`docs/containment_design.md` lists an escape test suite as a precondition for
turning execution on. `tests/test_containment_live.py` is that suite, but it
skips itself when no container runtime is up — which is the right default for a
test run and exactly the wrong thing for a precondition. A checklist item you
can satisfy by having Docker closed is not a checklist item.

So this script runs the same suite and treats a skip as a failure. It exits 0
only if every escape test actually started a container and the container held.

    python scripts/verify_containment.py
    python scripts/verify_containment.py --image python:3.11-slim

Run it on every host that will run an execution-enabled experiment, and again
after changing anything in controller/sandbox/. It takes a couple of minutes:
most of the tests deliberately wait out a timeout or fill a filesystem.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.config import RunConfig  # noqa: E402
from controller.sandbox.docker import DockerSandbox  # noqa: E402

SUITE = Path(__file__).parent.parent / "tests" / "test_containment_live.py"


def preflight(runtime: str, image: str) -> list[str]:
    """Reasons the suite could not run for real. Empty means good to go."""
    problems = []
    sb = DockerSandbox(runtime=runtime, image=image)

    if shutil.which(sb.runtime_argv[0]) is None:
        problems.append(f"{sb.runtime_argv[0]!r} is not on PATH.")
        return problems

    if not asyncio.run(sb.health_check()):
        problems.append(
            f"{sb.runtime_argv[0]!r} is installed but not responding. "
            f"Start the daemon (on macOS: open Docker Desktop) and retry."
        )
        return problems

    # A missing image would otherwise show up as a pile of SANDBOX_ERRORs that
    # look like containment failures.
    probe = subprocess.run(
        [*sb.runtime_argv, "image", "inspect", image],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        problems.append(
            f"Image {image!r} is not present locally. Pull it first:\n"
            f"    {sb.runtime_argv[0]} pull {image}\n"
            f"(The sandbox itself has no network, so it cannot pull at run time.)"
        )
    return problems


def report_capacity(runtime: str, memory: str) -> None:
    """What this host can actually back, which is not always what is configured."""
    sb = DockerSandbox(runtime=runtime, memory=memory)
    total = asyncio.run(sb.runtime_memory_bytes())
    gib = 1024 ** 3
    if total:
        print(f"Runtime memory available: {total / gib:.1f} GiB")
    print(f"sandbox_memory: {memory} — this bounds BOTH the container's memory "
          f"and its scratch space,\n  because /tmp is the only writable "
          f"filesystem and it is RAM-backed.")
    for problem in asyncio.run(sb.check_capacity()):
        print(f"\n  WARNING: {problem}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime", default="docker", help="docker | podman")
    ap.add_argument("--image", default="python:3.11-slim")
    ap.add_argument("--memory", default=RunConfig.model_fields["sandbox_memory"].default,
                    help="Check capacity for this sandbox_memory value")
    args = ap.parse_args()

    print(f"Verifying containment: runtime={args.runtime} image={args.image}\n")

    problems = preflight(args.runtime, args.image)
    if problems:
        print("Cannot verify containment:\n")
        for p in problems:
            print(f"  - {p}")
        print("\nUNVERIFIED. Do not enable execution on this host.")
        return 2

    report_capacity(args.runtime, args.memory)

    result = subprocess.run([
        sys.executable, "-m", "pytest", str(SUITE),
        "-m", "live_sandbox", "-v", "--no-header", "-p", "no:cacheprovider",
    ])

    # A skipped escape test proves nothing, so it cannot be allowed to pass.
    # Checked by re-collecting: pytest's exit code does not distinguish
    # "everything passed" from "everything skipped".
    collected = subprocess.run([
        sys.executable, "-m", "pytest", str(SUITE),
        "-m", "live_sandbox", "-q", "--collect-only", "-p", "no:cacheprovider",
    ], capture_output=True, text=True)
    n_tests = sum(1 for line in collected.stdout.splitlines() if "::" in line)

    print()
    if result.returncode != 0:
        print("CONTAINMENT FAILED. Do not enable execution on this host.")
        print("Read the failures above: each one names an escape that worked.")
        return 1
    if n_tests == 0:
        print("No escape tests ran. UNVERIFIED — do not enable execution.")
        return 2

    print(f"Containment verified on this host: {n_tests} escape tests, all held.")
    print("\nThe remaining checklist items in docs/containment_design.md are "
          "operational, not mechanical:")
    print("  - run the controller as a dedicated low-privilege OS user")
    print("  - verify host backups before the first execution-enabled run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
