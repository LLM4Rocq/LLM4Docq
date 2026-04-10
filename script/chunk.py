#!/usr/bin/env python3
"""Minimal chunker for Pile-of-Rocq exports.

This script prepares per-source chunks consumed by `compute_docstrings.py`.

Input format (one JSON object per line):
- `library` (str)
- `source.path` (str)
- `source.content` (str)
- `toc` (list)

Output format (one JSON object per line, per env):
- original fields
- `entry_key`
- `chunks` (list of `{chunk_id, chunk_doc, chunk_elements}`)
- `cache_field` (always present; copied if already available, else `{}`)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ANNOTATION_PATTERN = re.compile(r"^<annotation>(.*)</annotation>$")

TARGET_KINDS = {
    "definition",
    "syntactic_definition",
    "start_theorem_proof",
    "notation",
    "fixpoint",
    "cofixpoint",
    "coercion",
    "canonical",
    "instance",
    "inductive",
    "coinductive",
    "record",
    "structure",
    "variant",
    "class",
    "ltac",
    "constant",
    "field",
    "constructor",
    "hb",
}

PROOF_KINDS = {
    "proof",
    "proof_step",
    "subproof",
    "end_subproof",
    "bullet",
    "end_proof",
}

NOTATION_KINDS = {"notation", "syntactic_definition"}
UID_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


@dataclass
class ChunkConfig:
    prefix_size: int = 50
    max_lines: int = 100
    max_elements: int = 25


@dataclass
class EntryStats:
    targets: int = 0
    chunks: int = 0


@dataclass
class EnvStats:
    env: str
    entries: int = 0
    targets: int = 0
    chunks: int = 0


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_string_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key, doc in value.items():
        if isinstance(key, str) and isinstance(doc, str):
            out[key] = doc
    return out


def entry_key(entry: Mapping[str, Any]) -> str:
    library = str(entry.get("library", ""))
    source = entry.get("source")
    if not isinstance(source, Mapping):
        source_path = ""
        source_content = ""
    else:
        source_path = str(source.get("path", ""))
        source_content = str(source.get("content", ""))

    digest = _sha256_text(source_content)
    raw = f"{library}|{source_path}|{digest}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def span_offsets(element: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    span = element.get("span")
    if not isinstance(span, Mapping):
        return None
    bp = span.get("bp")
    ep = span.get("ep")
    if not isinstance(bp, int) or not isinstance(ep, int):
        return None
    if bp < 0 or ep < bp:
        return None
    return bp, ep


def remove_ocaml_comments(src: str, strict: bool = False) -> str:
    """Remove OCaml-style comments (* ... *) from source string."""
    lines: List[Tuple[str, bool]] = []
    buf: List[str] = []
    had_comment = False
    i, n = 0, len(src)
    depth = 0
    in_string = False
    in_char = False
    escape = False

    def flush_line() -> None:
        nonlocal buf, had_comment
        lines.append(("".join(buf), had_comment))
        buf = []
        had_comment = False

    while i < n:
        if src[i] == "\n":
            if depth > 0:
                had_comment = True
            else:
                buf.append("\n")
            flush_line()
            i += 1
            continue

        if src[i] == "\r":
            eol = "\r\n" if i + 1 < n and src[i + 1] == "\n" else "\r"
            if depth > 0:
                had_comment = True
            else:
                buf.append(eol)
            flush_line()
            i += len(eol)
            continue

        if depth > 0:
            if i + 1 < n and src[i : i + 2] == "(*":
                depth += 1
                i += 2
            elif i + 1 < n and src[i : i + 2] == "*)":
                depth -= 1
                i += 2
            else:
                i += 1
            had_comment = True
            continue

        ch = src[i]

        if in_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if in_char:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_char = False
            i += 1
            continue

        if i + 1 < n and src[i : i + 2] == "(*":
            depth = 1
            had_comment = True
            i += 2
        elif ch == '"':
            in_string = True
            buf.append(ch)
            i += 1
        elif ch == "'":
            in_char = True
            buf.append(ch)
            i += 1
        else:
            buf.append(ch)
            i += 1

    if strict and depth != 0:
        raise ValueError("Unclosed OCaml comment")

    if buf:
        lines.append(("".join(buf), had_comment))

    return "".join(text for (text, touched) in lines if not (touched and text.strip() == ""))


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    new_text = "\n".join(lines)
    while "\n\n\n" in new_text:
        new_text = new_text.replace("\n\n\n", "\n\n")
    return new_text


class RocqChunker:
    def __init__(self, cfg: ChunkConfig):
        self.cfg = cfg

    def extract_target_and_proof(
        self,
        toc: Sequence[Mapping[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        target_elements: List[Dict[str, Any]] = []
        proof_elements: List[Dict[str, Any]] = []
        namespaces_stack: List[Tuple[str, str]] = []

        for raw_entry in toc:
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            kind = entry.get("kind")
            name = entry.get("name")
            data = entry.get("data")
            if not isinstance(data, Mapping):
                data = {}
            data = dict(data)

            if kind == "begin_section":
                if isinstance(name, str) and name:
                    namespaces_stack.append(("SECTION", name))
            elif kind in {"declare_module_type", "define_module"}:
                is_alias = bool(data.get("is_alias", False))
                if not is_alias and isinstance(name, str) and name:
                    namespaces_stack.append(("MODULE", name))
            elif kind == "end_segment":
                if namespaces_stack:
                    namespaces_stack.pop()

            if kind in TARGET_KINDS:
                if not isinstance(name, str) or not name:
                    continue
                stack_modules = [n for scope, n in namespaces_stack if scope == "MODULE"]
                history = [n for _, n in namespaces_stack]
                fqn = ".".join(stack_modules + [name]) if stack_modules else name

                element = copy.deepcopy(entry)
                element_data = element.get("data")
                if not isinstance(element_data, Mapping):
                    element_data = {}
                element_data = dict(element_data)
                element_data["fqn"] = fqn
                element_data["history"] = ".".join(history)
                element["data"] = element_data
                target_elements.append(element)
            elif kind in PROOF_KINDS:
                proof_elements.append(copy.deepcopy(entry))

        return target_elements, proof_elements

    def _make_safe_uid(self, raw_uid: str, used: set[str]) -> str:
        safe = "".join(ch if ch in UID_SAFE_CHARS else "_" for ch in raw_uid)
        safe = safe.strip("._-")
        if not safe:
            safe = "uid"

        if safe not in used:
            return safe

        digest = hashlib.sha1(raw_uid.encode("utf-8")).hexdigest()[:8]
        candidate = f"{safe}_{digest}"
        if candidate not in used:
            return candidate

        idx = 1
        while True:
            candidate = f"{safe}_{digest}_{idx}"
            if candidate not in used:
                return candidate
            idx += 1

    def assign_uids(self, target_elements: List[Dict[str, Any]]) -> None:
        idx_notation = sum(1 for e in target_elements if e.get("kind") in NOTATION_KINDS)
        idx_uid: DefaultDict[str, int] = defaultdict(int)
        used_uids: set[str] = set()

        sorted_targets = sorted(
            target_elements,
            key=lambda e: span_offsets(e)[0] if span_offsets(e) else -1,
            reverse=True,
        )

        for element in sorted_targets:
            kind = element.get("kind")
            data = element.get("data")
            if not isinstance(data, Mapping):
                data = {}
            data = dict(data)
            fqn = str(data.get("fqn", "unknown"))

            if kind in NOTATION_KINDS:
                raw_uid = f"NOTATION_{idx_notation}"
                idx_notation -= 1
            elif idx_uid[fqn] > 0:
                raw_uid = f"{fqn}_duplicata_{idx_uid[fqn]}"
            else:
                raw_uid = fqn

            uid = self._make_safe_uid(raw_uid=raw_uid, used=used_uids)
            used_uids.add(uid)

            idx_uid[fqn] += 1
            data["uid"] = uid
            element["data"] = data

    def rewrite_source(
        self,
        source_content: str,
        target_elements: Sequence[Mapping[str, Any]],
        proof_elements: Sequence[Mapping[str, Any]],
    ) -> Tuple[str, str]:
        source_annotated_bytes = source_content.encode("utf-8")
        source_wo_proof_bytes = source_content.encode("utf-8")

        to_do: List[Tuple[int, str, Mapping[str, Any]]] = []
        for element in target_elements:
            offsets = span_offsets(element)
            if offsets is None:
                continue
            bp, _ = offsets
            to_do.append((bp, "COMMENT", element))

        for element in proof_elements:
            offsets = span_offsets(element)
            if offsets is None:
                continue
            bp, _ = offsets
            to_do.append((bp, "DELETE", element))

        to_do.sort(key=lambda x: x[0], reverse=True)

        for _, task, element in to_do:
            offsets = span_offsets(element)
            if offsets is None:
                continue
            bp, ep = offsets
            if task == "COMMENT":
                data = element.get("data")
                uid = data.get("uid") if isinstance(data, Mapping) else None
                if not isinstance(uid, str):
                    continue
                annotation = f"\n<annotation>{uid}</annotation>\n".encode("utf-8")
                source_annotated_bytes = (
                    source_annotated_bytes[:bp]
                    + annotation
                    + source_annotated_bytes[bp:ep]
                    + b"\n<END>\n"
                    + source_annotated_bytes[ep:]
                )
            elif task == "DELETE":
                source_annotated_bytes = source_annotated_bytes[:bp] + source_annotated_bytes[ep:]
                source_wo_proof_bytes = source_wo_proof_bytes[:bp] + source_wo_proof_bytes[ep:]

        src_annotated_str = remove_ocaml_comments(source_annotated_bytes.decode("utf-8"))
        src_wo_proof_str = source_wo_proof_bytes.decode("utf-8")

        return normalize_text(src_annotated_str), normalize_text(src_wo_proof_str)

    def chunk_doc(
        self,
        content: str,
        uid_map: Mapping[str, Mapping[str, Any]],
    ) -> List[Tuple[str, List[Mapping[str, Any]]]]:
        chunks: List[Tuple[str, List[Mapping[str, Any]]]] = []
        curr_prefix: List[str] = []
        curr_lines: List[str] = []
        curr_elements: List[Mapping[str, Any]] = []
        is_close = True

        for line in content.splitlines():
            match = ANNOTATION_PATTERN.match(line)
            should_flush = is_close and (
                len(curr_lines) >= self.cfg.max_lines
                or len(curr_elements) > self.cfg.max_elements
            )

            if should_flush:
                if curr_elements:
                    chunk_text = self._build_chunk_text(curr_prefix, curr_lines)
                    chunks.append((chunk_text, curr_elements))
                curr_prefix.extend([l for l in curr_lines if not l.startswith("<annotation>")])
                curr_lines = []
                curr_elements = []

            if match:
                uid = match.group(1)
                element = uid_map.get(uid)
                if element is not None:
                    curr_elements.append(element)
                curr_lines.append(line)
                is_close = False
            elif line == "<END>":
                is_close = True
            else:
                curr_lines.append(line)

        if curr_elements:
            chunk_text = self._build_chunk_text(curr_prefix, curr_lines)
            chunks.append((chunk_text, curr_elements))

        return chunks

    def _build_chunk_text(self, prefix: Sequence[str], lines: Sequence[str]) -> str:
        context_prefix = "\n".join(prefix[-self.cfg.prefix_size :])
        full_doc = "\n".join(lines)
        if context_prefix and full_doc:
            return f"{context_prefix}\n{full_doc}"
        return context_prefix or full_doc


def _is_target_toc_entry(entry: Mapping[str, Any]) -> bool:
    kind = entry.get("kind")
    name = entry.get("name")
    return bool(kind in TARGET_KINDS and isinstance(name, str) and name)


def annotate_toc_with_uids(
    toc: Sequence[Any],
    target_elements: Sequence[Mapping[str, Any]],
) -> List[Any]:
    out_toc: List[Any] = copy.deepcopy(list(toc))

    uid_sequence: List[str] = []
    for element in target_elements:
        data = element.get("data")
        uid = data.get("uid") if isinstance(data, Mapping) else None
        if isinstance(uid, str) and uid:
            uid_sequence.append(uid)

    uid_idx = 0
    for i, raw in enumerate(out_toc):
        if not isinstance(raw, Mapping) or not _is_target_toc_entry(raw):
            continue

        if uid_idx >= len(uid_sequence):
            logger.warning("TOC/target UID mismatch: missing generated UID for target #%d", uid_idx)
            break

        uid = uid_sequence[uid_idx]
        uid_idx += 1

        node = dict(raw)
        data = node.get("data")
        if not isinstance(data, Mapping):
            data = {}
        data = dict(data)
        data["uid"] = uid
        node["data"] = data
        out_toc[i] = node

    if uid_idx != len(uid_sequence):
        logger.warning(
            "TOC/target UID mismatch: assigned=%d generated=%d",
            uid_idx,
            len(uid_sequence),
        )

    return out_toc


def process_entry(chunker: RocqChunker, entry: Mapping[str, Any]) -> Tuple[Dict[str, Any], EntryStats]:
    library = entry.get("library")
    source = entry.get("source")
    toc = entry.get("toc")

    if not isinstance(library, str):
        raise ValueError("missing or invalid `library` field")
    if not isinstance(source, Mapping):
        raise ValueError("missing or invalid `source` field")
    if not isinstance(toc, list):
        raise ValueError("missing or invalid `toc` field")

    source_content = source.get("content")
    if not isinstance(source_content, str):
        raise ValueError("invalid `source.content`")

    target_elements, proof_elements = chunker.extract_target_and_proof(toc)
    chunker.assign_uids(target_elements)

    annotated_source, _ = chunker.rewrite_source(
        source_content=source_content,
        target_elements=target_elements,
        proof_elements=proof_elements,
    )

    uid_map: Dict[str, Mapping[str, Any]] = {}
    for element in target_elements:
        data = element.get("data")
        uid = data.get("uid") if isinstance(data, Mapping) else None
        if isinstance(uid, str):
            uid_map[uid] = element

    chunks_raw = chunker.chunk_doc(annotated_source, uid_map=uid_map)
    chunks: List[Dict[str, Any]] = []
    for chunk_id, (chunk_doc, elements) in enumerate(chunks_raw):
        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_doc": chunk_doc,
                "chunk_elements": elements,
            }
        )

    out = dict(entry)
    out["toc"] = annotate_toc_with_uids(toc=toc, target_elements=target_elements)
    out["entry_key"] = entry_key(entry)
    out["chunks"] = chunks
    out["cache_field"] = _as_string_map(entry.get("cache_field"))

    annotations = _as_string_map(entry.get("annotations"))
    if annotations:
        out["annotations"] = annotations

    stats = EntryStats(targets=len(target_elements), chunks=len(chunks))
    return out, stats


def iter_env_files(input_dir: Path, selected_envs: Optional[set[str]] = None) -> List[Path]:
    files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix in {".json", ".jsonl"}
    )
    if selected_envs is None:
        return files
    return [p for p in files if p.stem in selected_envs]


def process_env_file(env_file: Path, output_dir: Path, chunker: RocqChunker) -> EnvStats:
    env_name = env_file.stem
    output_file = output_dir / f"{env_name}.jsonl"
    stats = EnvStats(env=env_name)

    with env_file.open("r", encoding="utf-8") as file_in, output_file.open("w", encoding="utf-8") as file_out:
        for line_idx, line in enumerate(file_in, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if not isinstance(entry, Mapping):
                    raise ValueError("line is not a JSON object")
                transformed, entry_stats = process_entry(chunker, entry)
            except Exception as exc:
                raise RuntimeError(f"{env_file}:{line_idx}: {exc}") from exc

            file_out.write(json.dumps(transformed, ensure_ascii=False) + "\n")
            stats.entries += 1
            stats.targets += entry_stats.targets
            stats.chunks += entry_stats.chunks

    logger.info(
        "Processed %s -> %s (entries=%d targets=%d chunks=%d)",
        env_file.name,
        output_file.name,
        stats.entries,
        stats.targets,
        stats.chunks,
    )
    return stats

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build chunk_ter JSONL files for compute_docstrings.py from Pile-of-Rocq exports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("export"),
        help="Directory containing per-env JSON/JSONL export files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chunk_ter_output"),
        help="Directory where output per-env JSONL files are written.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment name (file stem). Repeat for multiple envs.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of env files processed in parallel.")
    parser.add_argument("--prefix-size", type=int, default=50, help="Max context lines kept before a chunk.")
    parser.add_argument("--max-lines", type=int, default=100, help="Max source lines per chunk window.")
    parser.add_argument("--max-elements", type=int, default=25, help="Max target elements per chunk window.")
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        logger.error("Input directory does not exist or is not a directory: %s", args.input_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_envs = set(args.env) if args.env else None
    env_files = iter_env_files(args.input_dir, selected_envs=selected_envs)
    if not env_files:
        logger.warning("No .json/.jsonl files found in %s", args.input_dir)
        return

    logger.info("Found %d env files in %s", len(env_files), args.input_dir)

    chunker = RocqChunker(
        ChunkConfig(
            prefix_size=args.prefix_size,
            max_lines=args.max_lines,
            max_elements=args.max_elements,
        )
    )

    all_stats: List[EnvStats] = []
    if args.workers <= 1:
        for env_file in tqdm(env_files, desc="Chunking env files"):
            all_stats.append(process_env_file(env_file=env_file, output_dir=args.output_dir, chunker=chunker))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_env_file, env_file=env_file, output_dir=args.output_dir, chunker=chunker)
                for env_file in env_files
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Chunking env files"):
                all_stats.append(future.result())

    total_entries = sum(s.entries for s in all_stats)
    total_targets = sum(s.targets for s in all_stats)
    total_chunks = sum(s.chunks for s in all_stats)

    logger.info(
        "Done. env_files=%d entries=%d targets=%d chunks=%d",
        len(all_stats),
        total_entries,
        total_targets,
        total_chunks,
    )


if __name__ == "__main__":
    main()
