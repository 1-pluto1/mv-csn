"""Google Drive bootstrap & persistence utilities for datasets, artifacts, fonts, and results.

Supports two seamless sync paths:
1. Native Google Colab Drive mount (`/content/drive/MyDrive/SYSU_DSE_MVCSN` via `google.colab.drive.mount`).
2. Local/headless `rclone` remote (`gdrive_pro:SYSU_DSE_MVCSN`) for automated CLI workflows.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_FOLDER_NAME = "SYSU_DSE_MVCSN"
DEFAULT_DRIVE_DIR = f"/content/drive/MyDrive/{DEFAULT_FOLDER_NAME}"


def _is_colab_runtime() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def mount_colab_drive(mount_point: str = "/content/drive", force_remount: bool = False) -> Optional[Path]:
    """Mount Google Drive inside an interactive Colab runtime if available."""
    mp = Path(mount_point)
    if (mp / "MyDrive").exists() and not force_remount:
        return mp / "MyDrive"
    if not _is_colab_runtime():
        return None
    try:
        from google.colab import drive  # type: ignore

        drive.mount(str(mp), force_remount=force_remount)
        mydrive = mp / "MyDrive"
        return mydrive if mydrive.exists() else None
    except Exception as exc:
        print(f"[gdrive_sync] Colab drive.mount skipped ({exc})", flush=True)
        return None


def resolve_drive_dir(
    drive_dir: str | Path | None = None,
    mount_if_colab: bool = False,
) -> Optional[Path]:
    """Resolve the target Drive workspace folder if mounted or configured."""
    if drive_dir is not None:
        return Path(drive_dir)
    env_dir = os.environ.get("MVCSN_GDRIVE_DIR")
    if env_dir:
        return Path(env_dir)
    if mount_if_colab and not Path("/content/drive/MyDrive").exists():
        mount_colab_drive("/content/drive")
    default_path = Path(DEFAULT_DRIVE_DIR)
    if default_path.parent.exists():
        return default_path
    return None


def _default_font_dir() -> Path:
    if Path("/content").exists() and os.access("/content", os.W_OK):
        return Path("/content/fonts")
    return Path("assets/fonts")


def _copy_tree_contents(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> int:
    if not src_dir.exists():
        return 0
    copied = 0
    for item in src_dir.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src_dir)
        dest = dst_dir / rel
        if dest.exists() and not overwrite:
            if dest.stat().st_size == item.stat().st_size:
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dest)
        copied += 1
    return copied


def bootstrap_from_gdrive(
    drive_dir: str | Path | None = None,
    repo_root: str | Path = ".",
    font_dir: str | Path | None = None,
    mount_if_colab: bool = False,
    overwrite: bool = False,
    rclone_remote: str | None = None,
) -> Dict[str, int]:
    """Restore cached resources, artifacts, fonts, and results from Google Drive into local workspace."""
    root = Path(repo_root)
    fdir = Path(font_dir) if font_dir is not None else _default_font_dir()
    counts = {"data/resources": 0, "artifacts": 0, "results": 0, "fonts": 0}

    if rclone_remote:
        remote = rclone_remote.rstrip("/")
        for sub, target in [
            ("data/resources", root / "data/resources"),
            ("artifacts", root / "artifacts"),
            ("results", root / "results"),
            ("fonts", fdir),
        ]:
            target.mkdir(parents=True, exist_ok=True)
            cmd = ["rclone", "copy", f"{remote}/{sub}", str(target)]
            if not overwrite:
                cmd.append("--ignore-existing")
            subprocess.run(cmd, check=True)
        return counts

    ddir = resolve_drive_dir(drive_dir=drive_dir, mount_if_colab=mount_if_colab)
    if ddir is None or not ddir.exists():
        return counts

    mapping = {
        "data/resources": root / "data/resources",
        "artifacts": root / "artifacts",
        "results": root / "results",
        "fonts": fdir,
    }
    for sub, local_dst in mapping.items():
        counts[sub] = _copy_tree_contents(ddir / sub, local_dst, overwrite=overwrite)
    print(f"[gdrive_sync] Bootstrapped from {ddir}: {counts}", flush=True)
    return counts


def persist_to_gdrive(
    drive_dir: str | Path | None = None,
    repo_root: str | Path = ".",
    font_dir: str | Path | None = None,
    mount_if_colab: bool = False,
    overwrite: bool = True,
    rclone_remote: str | None = None,
    subdirs: Iterable[str] = ("data/resources", "artifacts", "results", "fonts"),
) -> Dict[str, int]:
    """Persist workspace datasets, artifacts, results, and fonts back to Google Drive."""
    root = Path(repo_root)
    fdir = Path(font_dir) if font_dir is not None else _default_font_dir()
    counts: Dict[str, int] = {}

    if rclone_remote:
        remote = rclone_remote.rstrip("/")
        src_map = {
            "data/resources": root / "data/resources",
            "artifacts": root / "artifacts",
            "results": root / "results",
            "fonts": fdir,
        }
        for sub in subdirs:
            src = src_map[sub]
            if src.exists():
                subprocess.run(["rclone", "copy", str(src), f"{remote}/{sub}"], check=True)
        return counts

    ddir = resolve_drive_dir(drive_dir=drive_dir, mount_if_colab=mount_if_colab)
    if ddir is None:
        return counts
    ddir.mkdir(parents=True, exist_ok=True)

    src_map = {
        "data/resources": root / "data/resources",
        "artifacts": root / "artifacts",
        "results": root / "results",
        "fonts": fdir,
    }
    for sub in subdirs:
        counts[sub] = _copy_tree_contents(src_map[sub], ddir / sub, overwrite=overwrite)
    print(f"[gdrive_sync] Persisted to {ddir}: {counts}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync MV-CSN datasets, artifacts, fonts, and results with Google Drive.")
    parser.add_argument("--action", choices=["init", "save"], required=True)
    parser.add_argument("--drive-dir", default=None, help="Mounted Google Drive folder (default: /content/drive/MyDrive/SYSU_DSE_MVCSN)")
    parser.add_argument("--rclone-remote", default=None, help="Optional rclone remote path (e.g. gdrive_pro:SYSU_DSE_MVCSN)")
    parser.add_argument("--mount-if-colab", action="store_true", help="Attempt interactive google.colab.drive.mount")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.action == "init":
        bootstrap_from_gdrive(
            drive_dir=args.drive_dir,
            mount_if_colab=args.mount_if_colab,
            overwrite=args.overwrite,
            rclone_remote=args.rclone_remote,
        )
    else:
        persist_to_gdrive(
            drive_dir=args.drive_dir,
            mount_if_colab=args.mount_if_colab,
            overwrite=True,
            rclone_remote=args.rclone_remote,
        )


if __name__ == "__main__":
    main()
