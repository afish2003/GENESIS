# Containment Design — running agent-authored code safely

**Status**: proposal, not implemented
**Written**: 2026-09-12
**Applies to**: any future version where Axiom and Flux execute code rather than only write text

---

## 1. Why this document exists

GENESIS v1 has no sandbox, by explicit design. `PLAN.md` §4:

> All agent actions are mediated through the controller. Agents never hold a shell, a filesystem path outside the sealed world, or a network connection. The containment boundary is architectural and logical, not OS-level. There is no VM and no Docker container in version 1.

That holds *only because agents cannot execute anything*. The controller contains no `subprocess`, `os.system`, `exec`, `eval`, `compile`, `__import__`, or `popen`. Protocol documents are Markdown, serialized to JSON, and never run. The sole capability agents have is emitting text that the controller parses into Pydantic models.

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
| Memory cap | `--memory 2g --memory-swap 2g` | Prevents host OOM |
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

Config gains `sandbox_backend: null | docker` defaulting to `null`, and execution requires an explicit opt-in flag. Execution must never be reachable by default.

Every execution is logged as a first-class event — `CODE_EXECUTION` routed to its own JSONL — recording the code hash, entrypoint, exit code, truncated stdout/stderr, wall-clock time, and whether limits were hit. This is both audit trail and research data: *what the agents chose to build* is arguably the most interesting signal the platform could produce.

## 6. Preconditions before enabling execution

Every item must be true. None is optional.

- [ ] `ExecutionSandbox` implemented with `NullSandbox` as the default
- [ ] Container runs with: no network, read-only rootfs, non-root, all capabilities dropped, memory/CPU/PID/time caps
- [ ] Only `sandbox/workspace/` is mounted; verified by an in-container test asserting the repo, `.env`, `world/doctrine` and `world/memory` are unreachable
- [ ] Escape test suite — network egress, host filesystem reads, fork bomb, disk fill, infinite loop — each asserted to fail or be contained
- [ ] Timeout and kill path tested against a deliberately non-terminating program
- [ ] Output truncation tested against a program emitting unbounded stdout
- [ ] `CODE_EXECUTION` events logged, with limit-hit reasons
- [ ] Documented, reviewed recovery procedure for a container that will not die
- [ ] Run under a dedicated low-privilege OS user, not the researcher's account
- [ ] Host backups verified before the first execution-enabled run

## 7. Where this sits relative to the research plan

`PLAN.md` scopes v1 to protocol *documents* — text. Nothing in the current experimental design requires execution, and the BASELINE vs MEM_RESET question can be answered without it.

So this is a **v2 capability**, and the sequencing matters: run the pilot on the text-only design first. It will show whether the loop, the measures and the corpus hold up, and that is cheaper to learn before adding an isolation layer. Building containment now would be solving a problem the current experiment does not have.

What has changed today is the *other* half: the containment v1 actually needs — keeping model-supplied strings away from the filesystem — is no longer merely assumed. It is enforced and tested (`controller/world/paths.py`).
