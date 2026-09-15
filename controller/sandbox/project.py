"""Persistent, quota-capped project storage for agents that build real codebases.

The ephemeral workspace (`/workspace`, read-only, thrown away each execution) is
right for "write one module and run it". It is useless for "develop an
application over 100 cycles": nothing the agents write survives to the next
cycle, so there is no codebase to grow.

A persistent writable mount brings back the problem the read-only workspace was
closed to fix, and harder — a bind mount to the host filesystem has no size
quota, and now it is writable *and* long-lived. `--storage-opt size=` is not an
answer: on Docker Desktop's overlayfs driver it is accepted and silently
ignored (measured at 1600 MiB written under `size=1G`).

What does work is a filesystem that is genuinely that size: a fixed-size image,
formatted and loop-mounted, so the cap is enforced by the kernel's own ENOSPC
rather than by anything this code remembers to check. Measured: a 1 GiB image
stops a 4 GiB write at 900 MiB with errno 28.

Creating one needs privileges the controller must not have, so it is a
deliberate setup step — `scripts/setup_project_volume.py` — not something a run
does for itself. This module's job is to *refuse to use anything else*.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: A project filesystem may be at most this much larger than the size the run
#: asked for before it is assumed to be an ordinary directory on the host disk
#: rather than a dedicated capped volume. Filesystem overhead (ext4 metadata,
#: reserved blocks) means the usable size is a few percent *under* nominal, so
#: this only has to catch the case that matters: a 500 GiB host filesystem
#: standing in for a 32 GiB quota.
_SIZE_TOLERANCE = 1.25


class ProjectVolumeError(RuntimeError):
    """The configured project directory is not a safe place to let agents write."""


@dataclass(frozen=True)
class ProjectVolume:
    """A persistent directory agents may write to, and its enforced ceiling."""

    path: Path
    #: Total bytes of the filesystem the directory lives on, per statvfs.
    filesystem_bytes: int

    @property
    def free_bytes(self) -> int:
        st = os.statvfs(self.path)
        return st.f_bavail * st.f_frsize

    def describe(self) -> str:
        gib = 1024 ** 3
        return (f"{self.path} ({self.filesystem_bytes / gib:.1f} GiB filesystem, "
                f"{self.free_bytes / gib:.1f} GiB free)")


def parse_size(value: str) -> int:
    """'32g' / '512m' / '1.5t' / '2048' to bytes. 0 if unparseable."""
    text = str(value).strip().lower()
    units = {"b": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}
    multiplier = 1
    if text and text[-1] in units:
        multiplier = units[text[-1]]
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0


def _forbidden_ancestors(config) -> list[tuple[Path, str]]:
    """Paths the project volume must not be inside, and why.

    Agents writing into any of these would not merely be a containment problem;
    it would end the run's evidentiary value. Doctrine and memory are
    controller-mediated artifacts and the logs are append-only by design — an
    agent able to edit either can forge its own history, and no downstream
    analysis could tell.
    """
    repo = Path(__file__).resolve().parent.parent.parent
    return [
        (repo, "the GENESIS source tree — agents must not read the controller "
               "that measures them, quite apart from being able to edit it"),
        (Path(config.world_dir).resolve(), "the world directory — doctrine, "
                                           "memory and identity are "
                                           "controller-mediated artifacts"),
        (Path(config.research_logs_dir).resolve(), "the research logs — "
                                                   "append-only is the whole "
                                                   "basis of the audit trail"),
        (Path.home() / ".ssh", "your SSH keys"),
        (Path.home() / ".aws", "your cloud credentials"),
    ]


def resolve_project_volume(config) -> ProjectVolume | None:
    """Validate config.sandbox_project_dir, or None when no volume is configured.

    Raises rather than degrading: a project volume that is silently not mounted
    gives the agents an empty codebase every cycle and a run that looks like
    agents who cannot build anything. A project volume that is silently
    *uncapped* gives them the host's disk.
    """
    raw = getattr(config, "sandbox_project_dir", None)
    if raw is None:
        return None

    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise ProjectVolumeError(
            f"sandbox_project_dir {path} does not exist. Create it with:\n"
            f"    python scripts/setup_project_volume.py --path {path} "
            f"--size {config.sandbox_project_size}"
        )

    for ancestor, why in _forbidden_ancestors(config):
        try:
            resolved = ancestor.resolve()
        except OSError:
            continue
        if path == resolved or resolved in path.parents:
            raise ProjectVolumeError(
                f"Refusing to mount {path} as the agents' project directory: it "
                f"is inside {resolved}, which is {why}."
            )

    wanted = parse_size(config.sandbox_project_size)
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize

    if wanted and total > wanted * _SIZE_TOLERANCE:
        gib = 1024 ** 3
        raise ProjectVolumeError(
            f"Refusing to mount {path}: it is on a {total / gib:.1f} GiB "
            f"filesystem but sandbox_project_size is "
            f"{config.sandbox_project_size} ({wanted / gib:.1f} GiB). That is an "
            f"ordinary directory on the host disk, not a capped volume, so "
            f"nothing would stop agent code filling the disk — Docker has no "
            f"working size limit for a bind mount. Make a real one:\n"
            f"    python scripts/setup_project_volume.py --path {path} "
            f"--size {config.sandbox_project_size}\n"
            f"The check is on the FILESYSTEM's size, so a capped volume passes "
            f"and a directory on your root filesystem cannot."
        )

    if wanted and total < wanted * 0.5:
        # Not a safety problem — it is still capped, just smaller than the
        # researcher thinks. Worth saying, because "the agents ran out of space"
        # is a confusing thing to diagnose from evaluation scores.
        logger.warning(
            "sandbox_project_dir %s is a %.1f GiB filesystem but "
            "sandbox_project_size says %s. The agents have the smaller number.",
            path, total / 1024 ** 3, config.sandbox_project_size,
        )

    volume = ProjectVolume(path=path, filesystem_bytes=total)
    logger.info("Agents have a persistent project volume: %s", volume.describe())
    return volume
