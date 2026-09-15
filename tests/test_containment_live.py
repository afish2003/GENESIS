"""Escape tests that run real containers.

`tests/test_sandbox.py` asserts properties of the *command* — that `--network
none` is in the argv. This file asserts properties of the *container*: that a
socket actually fails to open. The two are not the same claim, and only the
second one is containment. A flag can be present and ineffective (wrong runtime,
a daemon-level override, a rootless quirk), and the argv test would still pass.

Every test here is skipped unless a container runtime is up, so the suite still
runs on a laptop with Docker closed. That makes it easy to believe containment
is verified when it has never run — so `scripts/verify_containment.py` runs the
same checks deliberately, prints what it found, and exits non-zero on failure.
That script is the precondition in docs/containment_design.md, not this file.

These are slow (a container start each) and marked `live_sandbox`:

    pytest tests/test_containment_live.py -m live_sandbox
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

from controller.sandbox.docker import DockerSandbox
from controller.sandbox.schemas import ExecutionOutcome, ExecutionRequest

pytestmark = pytest.mark.live_sandbox

TIMEOUT = 20.0


def _runtime_available() -> bool:
    sb = DockerSandbox()
    if shutil.which(sb.runtime_argv[0]) is None:
        return False
    return asyncio.run(sb.health_check())


requires_runtime = pytest.mark.skipif(
    not _runtime_available(),
    reason="no container runtime available (start Docker/Podman to run these)",
)


def run(code: str, *, timeout: float = TIMEOUT, entrypoint: str = "python module.py"):
    sb = DockerSandbox()
    return asyncio.run(sb.run(ExecutionRequest(
        files={"module.py": code}, entrypoint=entrypoint, timeout_seconds=timeout,
    )))


@requires_runtime
class TestTheSandboxWorksAtAll:
    """If these fail, every assertion below is vacuous."""

    def test_ordinary_code_runs_and_returns_its_output(self):
        r = run("print('hello from inside')")
        assert r.outcome is ExecutionOutcome.OK, r.detail
        assert "hello from inside" in r.stdout

    def test_a_crash_is_reported_not_raised(self):
        r = run("raise ValueError('boom')")
        assert r.outcome is ExecutionOutcome.NONZERO_EXIT
        assert "ValueError: boom" in r.stderr


@requires_runtime
class TestNetworkEgress:
    """--network none. Blocks exfiltration, code fetching, and reaching Ollama."""

    def test_outbound_tcp_fails(self):
        r = run(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=5)\n"
            "    print('ESCAPED: opened a socket')\n"
            "except OSError as e:\n"
            "    print('contained:', type(e).__name__)\n"
        )
        assert "ESCAPED" not in r.stdout, r.stdout
        assert "contained:" in r.stdout

    def test_dns_fails(self):
        r = run(
            "import socket\n"
            "try:\n"
            "    print('ESCAPED:', socket.gethostbyname('example.com'))\n"
            "except OSError as e:\n"
            "    print('contained:', type(e).__name__)\n"
        )
        assert "ESCAPED" not in r.stdout, r.stdout

    def test_the_inference_host_is_unreachable(self):
        """The one host on the LAN that would be interesting to reach."""
        r = run(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('192.168.1.100', 11434), timeout=5)\n"
            "    print('ESCAPED: reached Ollama')\n"
            "except OSError:\n"
            "    print('contained')\n"
        )
        assert "ESCAPED" not in r.stdout, r.stdout


@requires_runtime
class TestHostFilesystem:
    """Only the workspace is mounted, and it is read-only."""

    def test_the_repo_is_not_visible(self):
        r = run(
            "import os\n"
            "for p in ('/Users', '/home', '/host', '/repo'):\n"
            "    if os.path.isdir(p) and os.listdir(p):\n"
            "        print('ESCAPED:', p, os.listdir(p)[:5])\n"
            "print('checked')\n"
        )
        assert "ESCAPED" not in r.stdout, r.stdout

    def test_no_env_file_anywhere(self):
        r = run(
            "import os\n"
            "hits = []\n"
            "for root, dirs, files in os.walk('/'):\n"
            "    if '.env' in files:\n"
            "        hits.append(os.path.join(root, '.env'))\n"
            "    if len(hits) > 3:\n"
            "        break\n"
            "print('ESCAPED:' + str(hits) if hits else 'no .env reachable')\n"
        )
        assert "ESCAPED" not in r.stdout, r.stdout

    def test_the_root_filesystem_is_read_only(self):
        r = run(
            "try:\n"
            "    open('/etc/genesis_probe', 'w').write('x')\n"
            "    print('ESCAPED: wrote to /etc')\n"
            "except OSError as e:\n"
            "    print('contained:', type(e).__name__)\n"
        )
        assert "ESCAPED" not in r.stdout, r.stdout

    def test_the_workspace_is_read_only(self):
        """The bind mount is the only path to the host disk and has no quota,
        so a writable workspace is an uncapped disk-fill vector."""
        r = run(
            "try:\n"
            "    open('/workspace/probe', 'w').write('x')\n"
            "    print('ESCAPED: wrote to the workspace')\n"
            "except OSError as e:\n"
            "    print('contained:', type(e).__name__)\n"
        )
        assert "ESCAPED" not in r.stdout, r.stdout

    def test_running_as_nobody(self):
        r = run("import os; print('uid', os.getuid())")
        assert "uid 65534" in r.stdout, r.stdout


@requires_runtime
class TestResourceLimits:
    def test_disk_fill_is_capped(self):
        """tmpfs is the only writable filesystem and it is capped at 64 MiB."""
        r = run(
            "written = 0\n"
            "try:\n"
            "    with open('/tmp/fill', 'wb') as f:\n"
            "        for _ in range(4096):\n"          # 4 GiB if uncapped
            "            f.write(b'x' * 1024 * 1024)\n"
            "            f.flush()\n"
            "            written += 1\n"
            "    print('ESCAPED: wrote', written, 'MiB')\n"
            "except OSError as e:\n"
            "    print('contained after', written, 'MiB:', type(e).__name__)\n"
        )
        assert "ESCAPED" not in r.stdout, r.stdout

    def test_fork_bomb_is_contained(self):
        """--pids-limit. The container may die; the host must not."""
        r = run(
            "import os\n"
            "for _ in range(4096):\n"
            "    try:\n"
            "        if os.fork() == 0:\n"
            "            os._exit(0)\n"
            "    except OSError:\n"
            "        print('contained')\n"
            "        break\n"
            "else:\n"
            "    print('ESCAPED: forked 4096 times')\n"
        )
        assert "ESCAPED" not in r.stdout, r.stdout
        # And the runtime is still usable afterwards.
        assert run("print('still alive')").outcome is ExecutionOutcome.OK

    def test_memory_limit_is_enforced(self):
        r = run(
            "blocks = []\n"
            "try:\n"
            "    while True:\n"
            "        blocks.append(bytearray(32 * 1024 * 1024))\n"
            "except MemoryError:\n"
            "    print('contained: MemoryError')\n"
        )
        # Either Python raises MemoryError or the OOM killer takes it (137).
        assert r.outcome in (ExecutionOutcome.OK, ExecutionOutcome.TIMEOUT,
                             ExecutionOutcome.NONZERO_EXIT)
        assert "ESCAPED" not in r.stdout


@requires_runtime
class TestTheKillPath:
    """Checklist item 5: tested against a deliberately non-terminating program."""

    def test_an_infinite_loop_is_killed(self):
        r = run("while True:\n    pass\n", timeout=5.0)
        assert r.outcome is ExecutionOutcome.TIMEOUT, (r.outcome, r.detail)
        assert r.duration_seconds < 25, "the kill path did not fire promptly"

    def test_a_program_ignoring_sigterm_is_still_killed(self):
        """`timeout --signal=KILL` exists precisely for this case."""
        r = run(
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "while True:\n"
            "    time.sleep(0.1)\n",
            timeout=5.0,
        )
        assert r.outcome is ExecutionOutcome.TIMEOUT, (r.outcome, r.detail)

    def test_no_container_is_left_behind(self):
        """--rm plus the kill path. A survivor holds a workspace mount open."""
        import subprocess

        run("while True:\n    pass\n", timeout=5.0)
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        assert "genesis_" not in out, f"container left running:\n{out}"


@requires_runtime
class TestOutputFlooding:
    def test_a_chatty_program_cannot_flood_the_corpus(self):
        r = run("print('x' * 80)\n" * 1 + "for _ in range(200000): print('flood')")
        assert r.truncated is True
        assert len(r.stdout.encode()) < 100_000
