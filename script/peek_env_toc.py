#!/usr/bin/env python3
"""Quick navigator for env-level TOC + docstrings in exported HF parquet data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pq.read_table(path).to_pylist()


def _print_env_summary(env_dir: Path) -> None:
    meta_path = env_dir / "env_metadata.parquet"
    if not meta_path.exists():
        return
    row = pq.read_table(meta_path).to_pylist()[0]
    coverage = row.get("docstring_coverage")
    print("[env summary]")
    print(f"- sources: {row.get('source_rows')}")
    print(f"- toc rows: {row.get('toc_rows')}")
    print(f"- target toc rows: {row.get('target_toc_rows')}")
    print(f"- annotated toc rows: {row.get('annotated_toc_rows')}")
    if isinstance(coverage, float):
        print(f"- docstring coverage: {coverage:.4f}")
    else:
        print(f"- docstring coverage: {coverage}")
    print(f"- env toc rows: {row.get('env_toc_rows')}")


def _build_toc_indexes(env_toc_rows: List[Dict[str, Any]]):
    by_id: Dict[str, Dict[str, Any]] = {}
    children: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in env_toc_rows:
        node_id = _to_str(row.get("node_id"))
        if not isinstance(node_id, str):
            continue
        by_id[node_id] = row

    for row in env_toc_rows:
        parent_id = _to_str(row.get("parent_id"))
        if parent_id is None:
            parent_id = ""
        children[parent_id].append(row)

    for key in children:
        children[key].sort(key=lambda r: (_to_int(r.get("sibling_index")) or 0, _to_str(r.get("name")) or ""))

    roots: List[Dict[str, Any]] = []
    for row in env_toc_rows:
        parent_id = _to_str(row.get("parent_id"))
        if not isinstance(parent_id, str) or parent_id not in by_id:
            roots.append(row)
    roots.sort(key=lambda r: (_to_int(r.get("sibling_index")) or 0, _to_str(r.get("name")) or ""))

    return by_id, children, roots


def _describe_node(row: Dict[str, Any]) -> str:
    node_type = _to_str(row.get("node_type")) or "?"
    node_path = _to_str(row.get("path")) or _to_str(row.get("name")) or "?"
    n_children = _to_int(row.get("num_children")) or 0
    ann_count = _to_int(row.get("annotation_count"))
    ann_text = f", ann={ann_count}" if ann_count is not None else ""
    return f"[{node_type}] {node_path} (children={n_children}{ann_text})"


def _show_children(children_index: Dict[str, List[Dict[str, Any]]], parent_id: Optional[str], limit: int) -> None:
    key = parent_id if isinstance(parent_id, str) else ""
    rows = children_index.get(key, [])
    for row in rows[:limit]:
        print("- " + _describe_node(row))
    if len(rows) > limit:
        print(f"- ... ({len(rows) - limit} more)")


def _pick_file_row(env_toc_rows: List[Dict[str, Any]], file_query: str) -> Optional[Dict[str, Any]]:
    candidates = []
    for row in env_toc_rows:
        if _to_str(row.get("node_type")) != "file":
            continue
        source_path = _to_str(row.get("source_path_canonical"))
        node_path = _to_str(row.get("path"))
        if source_path == file_query or node_path == file_query:
            return row
        if (isinstance(source_path, str) and file_query in source_path) or (
            isinstance(node_path, str) and file_query in node_path
        ):
            candidates.append(row)

    if not candidates:
        return None
    candidates.sort(key=lambda r: _to_str(r.get("source_path_canonical")) or _to_str(r.get("path")) or "")
    if len(candidates) > 1:
        print(f"[file match] {len(candidates)} candidates, taking first:")
        for row in candidates[:5]:
            print("- " + (_to_str(row.get("source_path_canonical")) or _to_str(row.get("path")) or "?"))
    return candidates[0]


def _show_file_docstrings(env_dir: Path, file_row: Dict[str, Any], max_docstrings: int) -> None:
    source_id = _to_str(file_row.get("source_id"))
    source_path = _to_str(file_row.get("source_path_canonical")) or _to_str(file_row.get("path"))
    if not isinstance(source_id, str):
        print("[docstrings] file has no source_id match in env_toc; cannot inspect toc_nodes")
        return

    toc_nodes_path = env_dir / "toc_nodes.parquet"
    dataset = ds.dataset(str(toc_nodes_path), format="parquet")
    table = dataset.to_table(
        columns=["kind", "name", "docstring"],
        filter=ds.field("source_id") == source_id,
    )
    rows = table.to_pylist()

    doc_rows = [
        r
        for r in rows
        if isinstance(r.get("docstring"), str) and bool(r.get("docstring", "").strip())
    ]

    print("[docstrings]")
    print(f"- file: {source_path}")
    print(f"- source_id: {source_id}")
    print(f"- toc rows in file: {len(rows)}")
    print(f"- rows with docstring: {len(doc_rows)}")

    for row in doc_rows[:max_docstrings]:
        kind = _to_str(row.get("kind")) or "?"
        name = _to_str(row.get("name")) or ""
        doc = _to_str(row.get("docstring")) or ""
        doc_short = doc.replace("\n", " ").strip()
        if len(doc_short) > 180:
            doc_short = doc_short[:177] + "..."
        print(f"- [{kind}] {name}: {doc_short}")

    if len(doc_rows) > max_docstrings:
        print(f"- ... ({len(doc_rows) - max_docstrings} more docstrings)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse env_toc and inspect file docstrings.")
    parser.add_argument("--export-root", type=Path, default=Path("hf_export"), help="HF export root folder.")
    parser.add_argument("--env", required=True, help="Environment name (e.g. coq-actuary).")
    parser.add_argument(
        "--node-path",
        default=None,
        help="Substring to find env_toc node(s) by path, then list children.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="File path (exact or substring) to inspect docstrings (source_path_canonical or env_toc path).",
    )
    parser.add_argument("--max-children", type=int, default=25, help="Max children to display.")
    parser.add_argument("--max-docstrings", type=int, default=10, help="Max docstring samples to display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_dir = args.export_root / args.env
    if not env_dir.exists():
        raise FileNotFoundError(f"Missing env export directory: {env_dir}")

    env_toc_rows = _load_rows(env_dir / "env_toc.parquet")
    by_id, children_idx, roots = _build_toc_indexes(env_toc_rows)

    _print_env_summary(env_dir)

    print("[env_toc]")
    print(f"- rows: {len(env_toc_rows)}")

    if args.node_path:
        matches = [
            row
            for row in env_toc_rows
            if isinstance(_to_str(row.get("path")), str) and args.node_path in _to_str(row.get("path"))
        ]
        if not matches:
            print(f"- no node matches node-path={args.node_path!r}")
        else:
            matches.sort(key=lambda r: _to_str(r.get("path")) or "")
            print(f"- node-path matches: {len(matches)}")
            for row in matches[:10]:
                print("- " + _describe_node(row))

            chosen = matches[0]
            chosen_id = _to_str(chosen.get("node_id"))
            print("- children of first match:")
            _show_children(children_idx, chosen_id, args.max_children)
    else:
        print("- top-level nodes:")
        for row in roots[: args.max_children]:
            print("- " + _describe_node(row))
        if len(roots) > args.max_children:
            print(f"- ... ({len(roots) - args.max_children} more)")

    file_query = args.file
    if not file_query and args.node_path:
        # If node-path points directly to a unique file node, inspect it automatically.
        unique_file_matches = [
            row
            for row in env_toc_rows
            if _to_str(row.get("node_type")) == "file"
            and isinstance(_to_str(row.get("path")), str)
            and _to_str(row.get("path")) == args.node_path
        ]
        if len(unique_file_matches) == 1:
            file_query = _to_str(unique_file_matches[0].get("path"))

    if file_query:
        row = _pick_file_row(env_toc_rows, file_query)
        if row is None:
            print(f"[docstrings] no file matched query={file_query!r}")
            return
        _show_file_docstrings(env_dir, row, args.max_docstrings)


if __name__ == "__main__":
    main()
