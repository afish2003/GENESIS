# Containment Design — running agent-authored code safely

**Status**: wired and verified. The optional `execution` phase runs the cycle's artifact through the configured sandbox and logs `CODE_EXECUTION`. Both switches are off by default (`execution_enabled`, `sandbox_backend`), and the escape suite passes against real containers on the OptiPlex.
**Written**: 2026-09-12 · **Execution enabled**: 2026-09-14
**Applies to**: enabling execution on any host. Section 6 is the checklist; `scripts/verify_containment.py` mechanically checks the parts of it that can be.

---

## 1. Why this document exists

GENESIS v1 has no sandbox, by explicit design. `PLAN.md` §4:

> All agent actions are mediated through the controller. Agents never hold a shell, a filesystem path outside the sealed world, or a network connection. The containment boundary is architectural and logical, not OS-level. There is no VM and no Docker container in version 1.

That held *only because agents could not execute anything*, and it is no longer the description of the system. `controller/phases/execution.py` hands the cycle's artifact to a sandbox, and with `execution_enabled` plus `sandbox_backend=docker` the agents' code runs in a container. The boundary is now OS-level as well as architectural, which is what makes that acceptable — but `PLAN.md` §4 as written describes the default configuration, not this one.

*(Before 2026-09-13 this paragraph claimed the controller contained no subprocess call at all, which stopped being true when the sandbox was written. That sentence is the stated justification for `PLAN.md` §4's "no OS-level sandbox by design", so it is worth keeping accurate.)*

The moment that changes — agents writing code that actually runs — every assumption above dies at once. This document specifies what must exist first.

A caution about the current design's strength: it held in principle but not in practice. `protocol_id` was model-supplied and flowed unvalidated into a file path, letting a model write JSON anywhere the controller's user could write (fixed in `f69de9d`). An "architectural boundary" enforced by nothing but careful coding is one review lapse from failing. Execution raises the cost of such a lapse from *a stray file* to *arbitrary code as your user*.

## 2. Threat model

**Not assumed**: that the model is a capable adversary deliberately attacking the host. Qwen2.5 conducting research roleplay is not that.

**Assumed instead**, and sufficient to justify the work:

1. **Accident.** Generated code deletes the wrong directory, fills the disk, or loops forever. By far the most likely failure.
2. **Prompt-space pressure.** The experiment *deliberately applies adversarial pressure* — `doctrine_crisis_01`, `trust_test_01`, `identity_challenge_01` — and the codebook tracks "deception risk" and "self-protection" as measured variables. The design induces exactly the behaviour a sandbox defends against.
3. **Injected content.** Knowledge bases will be built from Wikipedia and arXiv (`PLAN.md` §9). Retrieved text enters agent context. A retrieved document containing instructions is a real injection vector once agents can act.
4. **Experimental validity.** This is the argument that matters most and is easiest to miss: if agents can touch anything outside the sealed world, **the sealed world premise is false and the run data is worthless**. Containment is not only host protection — it is a precondition for the results meaning anything. An escape does not merely risk the machine; it invalidates the experiment.

**Out of scope**: kernel 0-days, hardware side channels, a determined human attacker with local access.

## 3. Architecture

The invariant: **the controller never runs agent code, and agent code never sees the controller.**

```
OptiPlex (controller host)
│
├─ controller process          ← trusted; full user permissions
│    orchestrates 14 phases, owns world state and logs
│    NEVER executes agent output
│
├─ world/                      ← host-side source of truth
│    doctrine, memory, logs    ← never mounted into the sandbox
│    sandbox/workspace/        ← ONLY directory the container sees
│
└─ ephemeral container         ← untrusted; one per execution, --rm
     no network, read-only rootfs, non-root, resource-capped
     /workspace  (rw, bind)  ← the sole writable mount
     /tmp        (tmpfs, capped)

HP Omen — Ollama only. Agents never address it; the controller alone calls it.
```

Agent code is written into `world/sandbox/workspace/` by the controller, executed inside a fresh container, and results are read back from the same directory. The container is destroyed after each run.

### Required properties

| Property | Mechanism | Why |
|---|---|---|
| No network | `--network none` | Blocks exfiltration, code fetching, reaching Ollama or the LAN |
| No host filesystem | bind only `sandbox/workspace` | Doctrine, memory, logs, the repo and `.env` stay invisible |
| Non-root | `--user 65534:65534` | Limits damage on a container breakout |
| Immutable base | `--read-only` + tmpfs `/tmp` | Nothing persists between executions except the workspace |
| Memory cap | `--memory` == `--memory-swap`, `2g` by default (`config.sandbox_memory`) | Prevents host OOM; equal values disable swap. Also the scratch ceiling — see §9 |
| CPU cap | `--cpus 1.0` | Keeps the controller responsive |
| PID cap | `--pids-limit 128` | Stops fork bombs |
| Wall-clock cap | controller-side `asyncio.wait_for` + `docker kill` | Infinite loops end |
| Output cap | truncate stdout/stderr at a fixed size | Prevents log-flooding the corpus |
| No escape hatch | never mount the Docker socket, no `--privileged`, drop all capabilities | Mounting `/var/run/docker.sock` is instant host root |
| Ephemeral | `--rm`, fresh container per execution | No cross-cycle persistence the experiment cannot see |

### Deliberately excluded

- **Docker socket** — `-v /var/run/docker.sock` inside a container is equivalent to host root. Never.
- **The repo** — agents must not read GENESIS source. Beyond safety, an agent reading the controller that measures it is a confound.
- **`.env`** — holds API keys once hosted endpoints are in use.
- **`world/` above `sandbox/workspace/`** — doctrine and memory are controller-mediated artifacts. Direct write access would let agents forge their own history and destroy the audit trail.

## 4. Options considered

| Option | Isolation | Cost | Verdict |
|---|---|---|---|
| `subprocess` + `chroot`/rlimits | Weak | Low | **No.** Shared kernel, no network namespace, easy to get wrong. |
| **Docker/Podman, hardened** | Good | Low–medium | **Recommended.** Namespace isolation is sufficient for this threat model; tooling is standard; Podman rootless is strictly better where available. |
| gVisor (`runsc`) | Strong | Medium | Worth it only if untrusted third-party code is ever introduced. Syscall-interception overhead is irrelevant at this workload. |
| Firecracker / full VM | Strongest | High | Overkill. Real kernel isolation, but adds an image-management burden disproportionate to the risk. |

**Recommendation: hardened Podman (rootless) or Docker.** With `--network none`, a read-only rootfs, dropped capabilities and a non-root user, the remaining risk is a container-escape 0-day — far below the accident and validity risks this actually defends against.

## 5. Implementation sketch

A new backend behind an interface, mirroring how `InferenceBackend` is structured — so the sandbox is swappable and testable, and a `NullSandbox` can refuse execution outright as the default.

```
controller/sandbox/
    base.py       ExecutionSandbox ABC: run(files, entrypoint, timeout) -> ExecutionResult
    docker.py     DockerSandbox — the hardened runner above
    null.py       NullSandbox — refuses; the default, so execution is opt-in
    schemas.py    ExecutionRequest / ExecutionResult (Pydantic, logged like any artifact)
```

`config.sandbox_backend` (`null` | `docker`) defaults to `null`. There is currently **no CLI flag** — it is settable via `SANDBOX_BACKEND` or YAML only. Execution must never be reachable by default.

Every execution is logged as a first-class event — `CODE_EXECUTION` routed to its own JSONL — recording the code hash, entrypoint, exit code, truncated stdout/stderr, wall-clock time, and whether limits were hit. This is both audit trail and research data: *what the agents chose to build* is arguably the most interesting signal the platform could produce.

## 6. Preconditions before enabling execution

Every item must be true. None is optional. The mechanical ones are checked by
`scripts/verify_containment.py`, which runs `tests/test_containment_live.py`
against real containers and **treats a skip as a failure** — a checklist item you
can satisfy by having Docker closed is not a checklist item.

```bash
docker pull python:3.11-slim        # the sandbox has no network; pull first
python scripts/verify_containment.py
```

- [x] `ExecutionSandbox` implemented with `NullSandbox` as the default
- [x] Container runs with: no network, read-only rootfs, non-root, all capabilities dropped, memory/CPU/PID/time caps
- [x] Only the workspace is mounted, and read-only — asserted in the command (`tests/test_sandbox.py`) *and* inside a live container (`test_the_repo_is_not_visible`, `test_no_env_file_anywhere`, `test_the_workspace_is_read_only`)
- [x] Escape test suite — network egress, DNS, the Ollama host, host filesystem reads, `.env`, fork bomb, disk fill, memory exhaustion, infinite loop — each asserted to fail or be contained, in a real container
- [x] The claims the limits rest on are measured, not assumed: that tmpfs is charged to the memory cgroup, and that raising the cap raises usable scratch (§9)
- [x] Timeout and kill path tested against a deliberately non-terminating program, including one that ignores SIGTERM, with no container left behind afterwards
- [x] Output truncation tested against unbounded output
- [x] `CODE_EXECUTION` events logged, with limit-hit reasons (`ExecutionResult.limit_hit`, routed to `executions.jsonl`)
- [x] Documented recovery procedure for a container that will not die — section 8
- [ ] Run under a dedicated low-privilege OS user, not the researcher's account
- [ ] Host backups verified before the first execution-enabled run

The last two are operational and cannot be checked from inside the repo. They
are the two that remain open.

### Scratch space, and why it is bounded by memory

`/tmp` is the container's only writable filesystem, it is a tmpfs, and **tmpfs
pages are charged to the container's memory cgroup**. Measured, not assumed:
`--memory 512m --tmpfs /tmp:size=1g` writing 900 MiB is OOM-killed at exit 137.

So `size=` is not what contains a runaway write — `--memory` is. `/tmp` is
therefore sized to `sandbox_memory` rather than carrying its own smaller number,
which would have been a second, redundant limit whose only effect was to take
scratch away from the agents. `sandbox_memory` is the one knob:

| `sandbox_memory` | usable scratch |
|---|---|
| `512m` | ~300 MiB |
| `2g` (default) | ~1.7 GiB |
| `4g` | ~3.5 GiB |

It cannot usefully exceed what the runtime has — on Docker Desktop, the Linux
VM's allocation, not the Mac's. A `--memory` at or near the runtime total is not
a cap: the cgroup can never fire, so a runaway container takes the host down
instead of being killed. `check_capacity()` compares the two at run start, logs
it in `sandbox_health`, and `scripts/verify_containment.py` prints it.

**There is no disk-backed option.** `--storage-opt size=` is the obvious way to
get tens of GiB of scratch without spending RAM, and on the overlayfs driver
Docker Desktop uses it is *accepted and silently ignored* — measured at 1600 MiB
written under `size=1G`, with `df` reporting 911 GiB available. Using it would
reintroduce exactly the failure below: a flag that is present, correct, and does
nothing. `test_storage_opt_is_not_a_usable_alternative` pins that measurement and
fails if a future driver starts enforcing it, at which point disk-backed scratch
becomes worth designing.

If a task ever needs scratch on the order of tens of GiB, that is a v2 mechanism
— a host-prepared fixed-size filesystem image, or XFS project quotas on a Linux
host — not a bigger number in this config.

### What the escape suite actually found

The suite was written to confirm the design; it found a hole instead. The
workspace was bind-mounted `rw`, and a bind mount to the host filesystem **has
no size limit** — `--memory` caps RAM and `--tmpfs size=` caps `/tmp`, but
neither touches a bind mount. So `open("/workspace/big","w")` in a loop would
have filled the researcher's disk with every other containment property intact,
which is the single most likely accident in section 2's threat model.

The mount is now `ro`. Nothing is read back out of the workspace — results are
stdout and stderr — so this costs nothing, and `/tmp` is the only writable
filesystem in the container. The agents are told this in their design prompt.

This is the argument for running the suite on every host rather than trusting
the argv tests: `test_sandbox.py` asserted `-v ...:rw` was present and correct,
and it *was* present and correct. It was the property that was wrong.

## 7. Where this sits relative to the research plan

`PLAN.md` scopes v1 to protocol *documents* — text. Nothing in the BASELINE vs
MEM_RESET design requires execution, and that question can still be answered
without it. Execution is off by default for exactly that reason: the pilot
should run on the text-only design first, because the loop, the measures and the
corpus are cheaper to debug without an isolation layer underneath them.

What execution adds is the *other* half of the project's stated purpose — seeing
what the agents actually program, rather than what they typed. With it enabled:

- `correctness` is scored against what the code did, not how it reads
- `executions.jsonl` records every run: entrypoint, exit code, output, duration,
  which limit was hit
- `scripts/watch_run.py` shows the program running, inline, while the run happens

Turning it on is two settings, deliberately separate — `execution_enabled` adds
the phase, `sandbox_backend=docker` selects a runtime that is not a refusal:

```bash
EXECUTION_ENABLED=true SANDBOX_BACKEND=docker \
  python -m controller.main --run-id CODE_001 --condition BASELINE --cycles 20
```

## 8. Recovery — a container that will not die

The controller kills the container on the outer deadline and reports whether the
kill worked. When it did not, `ExecutionResult.limit_hit` is
`container_may_still_be_running`, the `execution_health` monitor rule raises a
CRITICAL anomaly, and `docker.py` logs at CRITICAL. The run continues, because
stopping it would not stop the container.

That state means a container is still holding CPU and a workspace mount. It has
no network and cannot write anywhere that persists, so it is a resource problem
rather than a containment failure — but it will not clean itself up.

```bash
# 1. Find it. Sandbox containers are always named genesis_genesis_ws_*
docker ps --filter "name=genesis_" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

# 2. Kill it. SIGKILL; there is nothing inside worth draining.
docker kill $(docker ps -q --filter "name=genesis_")

# 3. If `docker kill` hangs, the daemon is the problem, not the container.
docker ps           # if this also hangs, restart the daemon:
#   macOS:  killall Docker && open -a Docker
#   Linux:  sudo systemctl restart docker

# 4. Anything left behind. --rm should make this empty.
docker ps -a --filter "name=genesis_" --format "{{.Names}}"
docker rm -f $(docker ps -aq --filter "name=genesis_")   # if not

# 5. Orphaned workspaces, if the controller was killed before its finally block.
ls -d ${TMPDIR:-/tmp}/genesis_ws_* 2>/dev/null && rm -rf ${TMPDIR:-/tmp}/genesis_ws_*
```

Then decide about the run. The cycle itself is fine — the execution came back as
a TIMEOUT and everything downstream continued — so the data is usable. What is
worth checking before continuing is whether the daemon is healthy
(`python scripts/verify_containment.py`), because if it is not, subsequent
cycles will record SANDBOX_ERROR and their `correctness` scores will be measuring
the host rather than the agents.

**If a container ever appears that you cannot account for** — a name that is not
`genesis_ws_*`, a published port, a mount you did not configure — treat it as a
containment failure rather than a resource leak: stop the run, keep the logs,
and do not restart with execution enabled until `verify_containment.py` passes
and you know where it came from.


## 9. Persistent project storage

The ephemeral read-only workspace is right for "write one module and run it" and
useless for "develop an application over 100 cycles": nothing survives to the
next cycle, so there is no codebase to grow. `sandbox_project_dir` mounts a
persistent writable directory at `/project`, which also becomes the working
directory when code runs.

That is the one long-lived writable thing the agents get, so the containment
question is sharp: **a Docker bind mount has no size limit, and Docker cannot
give it one.** `--storage-opt size=` is accepted and silently ignored on the
overlayfs driver (§6). So the cap cannot come from Docker. It comes from the
directory being a filesystem that is genuinely that size, and the controller
**refuses to mount anything else** — it compares `statvfs` against
`sandbox_project_size` and rejects an ordinary directory with instructions
rather than running with an uncapped path to the host disk.

```bash
python scripts/setup_project_volume.py --path ~/genesis_project --size 32g
```

| Host | Mechanism | Privilege |
|---|---|---|
| Linux | sparse image + `mkfs.ext4` + `mount -o loop` | one `sudo` at setup |
| macOS | `hdiutil` APFS sparse bundle | none |

Both are sparse: a 32 GiB volume costs ~25 MB until the agents fill it. Both
were measured to produce ENOSPC at the cap from inside a container — 200 MiB
written to a 300 MiB macOS bundle, errno 28; 900 MiB to a 1 GiB ext4 loop image,
errno 28.

Also refused, regardless of size: any path inside the repo, `world/`, the
research logs, `~/.ssh` or `~/.aws`. Doctrine and memory are controller-mediated
artifacts and the logs are append-only by design; an agent able to write either
can forge its own history, and no downstream analysis could tell.

Running out of space is a bad cycle, not a broken run — the execution reports a
non-zero exit with the ENOSPC traceback, the evaluator sees it, and the next
cycle proceeds.

### What this changes about the experiment

Worth stating plainly, because it is easy to miss: **the project volume is a
second channel of persistence that the controller does not mediate.** Under
MEM_RESET the agents' journals and self-history are wiped and their codebase is
not. That is a legitimate and interesting design — agents who forget what they
decided but inherit what they built — but it is not what MEM_RESET meant before,
and any arm using both should say so.

## 10. Sizing, in one place

The only number most runs need to change:

```yaml
sandbox_memory: 4g            # container memory AND /tmp scratch; see section 6
sandbox_project_dir: ~/genesis_project   # persistent codebase; section 9
sandbox_project_size: 32g     # checked against the real filesystem, not trusted
```

Check it against the host before a run that matters:

```bash
python scripts/verify_containment.py --memory 4g
```

It prints what the runtime actually has and warns if the cap exceeds 75% of it,
because past that point the cgroup stops being the thing that fires. On Docker
Desktop, raise the VM's allocation in Settings > Resources first; the Mac's
total RAM is not what containers get.
