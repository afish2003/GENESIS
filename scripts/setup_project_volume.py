"""Create a fixed-size filesystem for agents to build in.

A bind mount to an ordinary host directory has NO size limit. Docker's
`--storage-opt size=` looks like the fix and is accepted-and-ignored on the
overlayfs driver Docker Desktop uses — measured at 1600 MiB written under
`size=1G`. So the only cap that actually holds is a filesystem that is genuinely
that size: an image file, formatted, and loop-mounted. Then a runaway write ends
in ENOSPC from the kernel, not from anything this project remembered to check.

    python scripts/setup_project_volume.py --path ~/genesis_project --size 32g

This needs privileges the controller deliberately does not have, which is why it
is a separate step you run once rather than something a run does for itself.
Linux gets a loop-mounted ext4 image (one sudo, to mount). macOS gets an
hdiutil sparse bundle (no sudo at all). Both are sparse: a 32 GiB volume costs
almost nothing until the agents fill it.

The mount does not survive a reboot. Re-run this — it reuses the existing image
and only remounts — or use the printed fstab/hdiutil line.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.sandbox.project import parse_size  # noqa: E402

GIB = 1024 ** 3


def is_mounted(path: Path) -> bool:
    """Whether `path` is a mount point — i.e. its own filesystem, i.e. capped."""
    return path.is_dir() and os.path.ismount(path)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, **kw)


def create_image(image: Path, size_bytes: int) -> None:
    if image.exists():
        actual = image.stat().st_size
        print(f"Image already exists: {image} ({actual / GIB:.1f} GiB)")
        if abs(actual - size_bytes) > size_bytes * 0.01:
            print(f"  NOTE: it is not the {size_bytes / GIB:.1f} GiB you asked "
                  f"for. Delete it to resize; resizing in place is not done "
                  f"here because it can lose data.")
        return

    image.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(image.parent).free
    if size_bytes > free:
        raise SystemExit(
            f"Cannot create a {size_bytes / GIB:.1f} GiB image: only "
            f"{free / GIB:.1f} GiB free on {image.parent}."
        )
    print(f"Creating {size_bytes / GIB:.1f} GiB image at {image}")
    with open(image, "wb") as f:      # sparse: allocates as it is used
        f.truncate(size_bytes)


def format_and_mount_linux(image: Path, mount: Path) -> None:
    mount.mkdir(parents=True, exist_ok=True)
    if is_mounted(mount):
        print(f"Already mounted: {mount}")
        return
    if run(["file", "-b", str(image)], capture_output=True, text=True
           ).stdout.find("ext") < 0:
        run(["mkfs.ext4", "-q", "-F", str(image)], check=True)
    print("Mounting (sudo — mounting a loop device needs root):")
    run(["sudo", "mount", "-o", "loop", str(image), str(mount)], check=True)
    # Agents run as uid 65534 inside the container; the directory must be
    # writable by that uid on the host side of the bind mount.
    run(["sudo", "chown", "65534:65534", str(mount)], check=True)
    print(f"\nTo mount at boot, add to /etc/fstab:\n"
          f"  {image}  {mount}  ext4  loop,nofail  0  0")


def format_and_mount_macos(image: Path, mount: Path, size_bytes: int) -> None:
    """macOS: a sparse disk image, attached with hdiutil. No sudo needed.

    Docker Desktop reaches the Mac filesystem over virtiofs, which has no quota
    of its own — but it faithfully passes through the underlying filesystem's
    ENOSPC. So an image that is genuinely 32 GiB caps a container writing
    through the bind mount, measured: a 300 MiB volume stops a 1 GiB write at
    200 MiB with errno 28.

    SPARSEBUNDLE rather than a flat .dmg so the file grows as it is used: a
    32 GiB project volume costs ~25 MB until the agents put something in it.
    """
    mount.mkdir(parents=True, exist_ok=True)
    if is_mounted(mount):
        print(f"Already mounted: {mount}")
        return

    bundle = image.with_suffix(".sparsebundle")
    if not bundle.exists():
        print(f"Creating {size_bytes / GIB:.1f} GiB sparse image at {bundle}")
        run(["hdiutil", "create", "-size", str(size_bytes // (1024 * 1024)) + "m",
             "-fs", "APFS", "-volname", mount.name, "-type", "SPARSEBUNDLE",
             "-quiet", str(bundle.with_suffix(""))], check=True)

    run(["hdiutil", "attach", str(bundle), "-mountpoint", str(mount),
         "-nobrowse", "-quiet"], check=True)
    # The container runs as uid 65534; APFS volumes mount with ownership
    # enabled, so the mount point has to be writable by it.
    os.chmod(mount, 0o777)
    print(f"\nTo remount after a reboot:\n"
          f"  hdiutil attach {bundle} -mountpoint {mount} -nobrowse")
    print(f"To unmount:\n  hdiutil detach {mount}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True,
                    help="Where to mount it — this becomes sandbox_project_dir")
    ap.add_argument("--size", default="32g", help="e.g. 32g, 100g")
    ap.add_argument("--image", default=None,
                    help="Backing image file (default: <path>.img)")
    args = ap.parse_args()

    mount = Path(args.path).expanduser().resolve()
    image = Path(args.image).expanduser().resolve() if args.image else \
        mount.with_suffix(".img")
    size_bytes = parse_size(args.size)
    if not size_bytes:
        raise SystemExit(f"Could not parse --size {args.size!r}")

    if platform.system() == "Linux":
        create_image(image, size_bytes)
        format_and_mount_linux(image, mount)
    elif platform.system() == "Darwin":
        format_and_mount_macos(image, mount, size_bytes)
    else:
        raise SystemExit(
            f"No project-volume recipe for {platform.system()}. The requirement "
            f"is a filesystem that is genuinely the configured size; create one "
            f"however this OS does that, mount it at --path, and the controller "
            f"will accept it. It checks the filesystem, not the method."
        )

    if not is_mounted(mount):
        print(f"\n{mount} is not a mount point. The controller will refuse it.")
        return 1

    st = os.statvfs(mount)
    print(f"\nReady: {mount}")
    print(f"  filesystem {st.f_blocks * st.f_frsize / GIB:.1f} GiB, "
          f"{st.f_bavail * st.f_frsize / GIB:.1f} GiB free")
    print(f"\nUse it:\n  SANDBOX_PROJECT_DIR={mount}\n  SANDBOX_PROJECT_SIZE={args.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
