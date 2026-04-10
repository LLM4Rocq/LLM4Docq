#!/usr/bin/env python3
"""
Build a lightweight legacy cache from export_v2.

Copies only:
- one_liner.txt (if present)
- header.txt (if present)
- source.sha256 (always for copied folders)

Directory layout is preserved, so compute_docstrings can load it directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_folder_identity(root: Path, folder: Path) -> Tuple[str, str, str]:
    rel = folder.relative_to(root)
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"invalid folder structure: {folder}")
    env = parts[0]
    library = parts[1]
    rel_no_ext = PurePosixPath(*parts[2:]).as_posix()
    return env, library, rel_no_ext


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a minimal legacy text cache for HPC.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("export_v2/export/theostos"),
        help="Root folder of the legacy export.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("export_v2_text_cache"),
        help="Destination folder for the minimal cache.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path (default: <output-root>/manifest.jsonl).",
    )
    args = parser.parse_args()

    if not args.input_root.exists() or not args.input_root.is_dir():
        raise SystemExit(f"Invalid input root: {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.output_root / "manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    copied_folders = 0
    copied_headers = 0
    copied_one_liners = 0
    manifest_rows: List[Dict[str, object]] = []

    source_files = sorted(args.input_root.glob("*/*/**/source.v"))
    logger.info("Scanning %d source folders...", len(source_files))

    for source_file in source_files:
        folder = source_file.parent
        one_liner_src = folder / "one_liner.txt"
        header_src = folder / "header.txt"
        has_one_liner = one_liner_src.exists()
        has_header = header_src.exists()
        if not has_one_liner and not has_header:
            continue

        rel_folder = folder.relative_to(args.input_root)
        out_folder = args.output_root / rel_folder
        out_folder.mkdir(parents=True, exist_ok=True)

        if has_one_liner:
            shutil.copy2(one_liner_src, out_folder / "one_liner.txt")
            copied_one_liners += 1
        if has_header:
            shutil.copy2(header_src, out_folder / "header.txt")
            copied_headers += 1

        digest = sha256_file(source_file)
        (out_folder / "source.sha256").write_text(digest, encoding="utf-8")

        env, library, rel_no_ext = parse_folder_identity(args.input_root, folder)
        manifest_rows.append(
            {
                "env": env,
                "library": library,
                "rel_no_ext": rel_no_ext,
                "source_sha256": digest,
                "has_one_liner": has_one_liner,
                "has_header": has_header,
            }
        )
        copied_folders += 1

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        (
            "Done. copied_folders=%d one_liner=%d header=%d manifest=%s output_root=%s"
        ),
        copied_folders,
        copied_one_liners,
        copied_headers,
        manifest_path,
        args.output_root,
    )


if __name__ == "__main__":
    main()
