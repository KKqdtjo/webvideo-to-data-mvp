"""Verify that a built wheel retains every pinned Panda source asset."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import PurePosixPath
from zipfile import ZipFile


_ASSET_SUFFIX = "webvideo_to_data/assets/mujoco_menagerie/franka_emika_panda"


def _asset_root_member(names: list[str]) -> str:
    candidates = [name for name in names if name.endswith(f"{_ASSET_SUFFIX}/UPSTREAM.json")]
    if len(candidates) != 1:
        raise ValueError("wheel must contain exactly one Panda UPSTREAM.json")
    return str(PurePosixPath(candidates[0]).parent)


def verify_wheel_assets(wheel_path: str) -> None:
    with ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        asset_root = _asset_root_member(names)
        metadata = json.loads(wheel.read(f"{asset_root}/UPSTREAM.json"))
        for required in (
            "LICENSE",
            "README.md",
            "CHANGELOG.md",
            "panda_exp001.xml",
            "exp001_scene.xml",
        ):
            if f"{asset_root}/{required}" not in names:
                raise ValueError(f"wheel is missing required Panda asset: {required}")
        upstream_files = metadata.get("upstream_files")
        if not isinstance(upstream_files, dict):
            raise ValueError("UPSTREAM.json must contain an upstream_files mapping")
        for relative_name, expected_hash in upstream_files.items():
            if not isinstance(relative_name, str) or not isinstance(expected_hash, str):
                raise ValueError("UPSTREAM.json upstream_files entries must be string pairs")
            member_name = f"{asset_root}/{relative_name}"
            if member_name not in names:
                raise ValueError(f"wheel is missing pinned upstream asset: {relative_name}")
            actual_hash = hashlib.sha256(wheel.read(member_name)).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"wheel hash mismatch for {relative_name}: {actual_hash} != {expected_hash}"
                )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_wheel_assets.py WHEEL_PATH")
    verify_wheel_assets(sys.argv[1])
