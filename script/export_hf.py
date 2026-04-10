#!/usr/bin/env python3
"""Export Pile-of-Rocq dumps to Hugging Face-ready Parquet tables.

This exporter is designed for the LLM4Docq workspace:
- base source dumps: `../Pile-of-rocq/export_v2/*.jsonl`
- env configs: `../Pile-of-rocq/config/*.yaml`
- merged docstrings: `exported_v3/merged/*.jsonl`
- merged env TOC trees: `exported_v3/merged/toc/*.toc.json`

For each environment it writes normalized parquet tables, with:
- source TOC nodes enriched with a single `docstring` field
- env-level TOC exported as one `env_toc.parquet`

Optional push-to-hub supports retries and resumable large-folder uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, DefaultDict, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover
    HfApi = None


SCHEMA_VERSION = 4
BASE_TABLES_ORDER = [
    "sources",
    "toc_nodes",
    "proofs",
    "proof_steps",
    "step_deps",
    "proof_axioms",
    "env_toc",
    "env_metadata",
]

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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _to_string_list(value: Any) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, str):
            out.append(item)
        else:
            out.append(_to_str(item) or _json_compact(item))
    return out


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _prefer_longer_string(old: Any, new: Any) -> Optional[str]:
    if _is_non_empty_string(old) and _is_non_empty_string(new):
        return old if len(old.strip()) >= len(new.strip()) else new
    if _is_non_empty_string(old):
        return old
    if _is_non_empty_string(new):
        return new
    return None


def _normalize_env_id(name: str) -> str:
    return name.strip()


def _parse_range(range_obj: Any) -> Dict[str, Optional[int]]:
    if not isinstance(range_obj, dict):
        return {
            "start_line": None,
            "start_character": None,
            "end_line": None,
            "end_character": None,
        }
    start = range_obj.get("start") or {}
    end = range_obj.get("end") or {}
    if not isinstance(start, dict):
        start = {}
    if not isinstance(end, dict):
        end = {}
    return {
        "start_line": _to_int(start.get("line")),
        "start_character": _to_int(start.get("character")),
        "end_line": _to_int(end.get("line")),
        "end_character": _to_int(end.get("character")),
    }


def _parse_span(span_obj: Any) -> Dict[str, Optional[int]]:
    if not isinstance(span_obj, dict):
        return {"bp": None, "ep": None}
    return {
        "bp": _to_int(span_obj.get("bp")),
        "ep": _to_int(span_obj.get("ep")),
    }


def _source_relative_path(source_path: str, library: str) -> str:
    normalized = source_path.replace("\\", "/")
    marker = f"user-contrib/{library}/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    if library == "Corelib":
        theory_marker = "/theories/"
        if theory_marker in normalized:
            return normalized.split(theory_marker, 1)[1]
        dup_marker = "/user-contrib/Corelib_duplicate/"
        if dup_marker in normalized:
            return normalized.split(dup_marker, 1)[1]
    return Path(normalized).name


def _source_relative_without_ext(source_path: str, library: str) -> str:
    rel = _source_relative_path(source_path, library)
    rel_path = PurePosixPath(rel)
    if rel_path.suffix:
        rel_path = rel_path.with_suffix("")
    return rel_path.as_posix()


def _source_canonical_path(source_path: str, library: str) -> str:
    rel = _source_relative_path(source_path, library)
    if library == "Corelib":
        return f"Corelib/{rel}"
    return f"{library}/{rel}"


def _source_identity(library: str, source_path: str, content: str) -> Tuple[str, str, str]:
    canonical_path = _source_canonical_path(source_path, library)
    content_sha = _sha256_text(content)
    source_id = _sha256_text(f"{library}:{canonical_path}:{content_sha}")
    return source_id, content_sha, canonical_path


def _entry_key_from_source(library: str, source_path: str, source_content: str) -> str:
    digest = _sha256_text(source_content)
    raw = f"{library}|{source_path}|{digest}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any], str]]:
    with path.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            raw = line.rstrip("\n")
            if not raw:
                continue
            yield i, json.loads(raw), raw


def _normalize_children(children: Any) -> List[str]:
    if not isinstance(children, Sequence) or isinstance(children, (str, bytes, bytearray)):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in children:
        if not isinstance(item, str) or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_locations(locations: Any) -> List[Dict[str, Optional[int]]]:
    if not isinstance(locations, Sequence) or isinstance(locations, (str, bytes, bytearray)):
        return []
    out: List[Dict[str, Optional[int]]] = []
    for raw in locations:
        if not isinstance(raw, Mapping):
            continue
        rng = _parse_range(raw.get("range"))
        out.append(
            {
                "range_start_line": rng["start_line"],
                "range_start_character": rng["start_character"],
                "range_end_line": rng["end_line"],
                "range_end_character": rng["end_character"],
            }
        )
    return out


def _normalize_symbol_refs(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return []
    out: List[Dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        out.append(
            {
                "name": _to_str(raw.get("name")),
                "logical_path": _to_str(raw.get("logical_path")),
                "physical_path": _to_str(raw.get("physical_path")),
                "locations": _normalize_locations(raw.get("locations")),
            }
        )
    return out


def _normalize_goal_hyps(hyps: Any) -> List[Dict[str, Any]]:
    if not isinstance(hyps, Sequence) or isinstance(hyps, (str, bytes, bytearray)):
        return []
    out: List[Dict[str, Any]] = []
    for raw in hyps:
        if not isinstance(raw, Mapping):
            continue
        out.append(
            {
                "names": _to_string_list(raw.get("names")),
                "ty": _to_str(raw.get("ty")),
            }
        )
    return out


def _normalize_goals(goals: Any) -> List[Dict[str, Any]]:
    if not isinstance(goals, Sequence) or isinstance(goals, (str, bytes, bytearray)):
        return []
    out: List[Dict[str, Any]] = []
    for raw in goals:
        if not isinstance(raw, Mapping):
            continue
        out.append(
            {
                "evar": _to_int(raw.get("evar")),
                "name": _to_str(raw.get("name")),
                "hyps": _normalize_goal_hyps(raw.get("hyps")),
                "ty": _to_str(raw.get("ty")),
            }
        )
    return out


def _normalize_goal_state(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return {
            "goals": [],
            "stack_json": [],
            "bullet": None,
            "shelf_json": [],
            "given_up_json": [],
        }
    return {
        "goals": _normalize_goals(value.get("goals")),
        "stack_json": [_json_compact(x) for x in (value.get("stack") if isinstance(value.get("stack"), Sequence) and not isinstance(value.get("stack"), (str, bytes, bytearray)) else [])],
        "bullet": _to_str(value.get("bullet")),
        "shelf_json": [_json_compact(x) for x in (value.get("shelf") if isinstance(value.get("shelf"), Sequence) and not isinstance(value.get("shelf"), (str, bytes, bytearray)) else [])],
        "given_up_json": [_json_compact(x) for x in (value.get("given_up") if isinstance(value.get("given_up"), Sequence) and not isinstance(value.get("given_up"), (str, bytes, bytearray)) else [])],
    }


def _extract_toc_data_fields(data: Any) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[bool], Optional[str]]:
    if not isinstance(data, Mapping):
        return None, None, None, None, None
    uid = _to_str(data.get("uid"))
    fqn = _to_str(data.get("fqn"))
    history = _to_str(data.get("history"))
    is_alias_raw = data.get("is_alias")
    is_alias = bool(is_alias_raw) if isinstance(is_alias_raw, bool) else None
    extra = {
        k: v
        for k, v in data.items()
        if k not in {"uid", "fqn", "history", "is_alias"}
    }
    extra_json = _json_compact(extra) if extra else None
    return uid, fqn, history, is_alias, extra_json


def _normalize_targets(targets: Any) -> List[Dict[str, Any]]:
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes, bytearray)):
        return []
    out: List[Dict[str, Any]] = []
    for raw in targets:
        if not isinstance(raw, Mapping):
            continue
        extra = {
            k: v
            for k, v in raw.items()
            if k
            not in {
                "lib",
                "copy_skeleton",
                "packages",
                "extra_coq_proj_args",
                "extra_coq_makefile_args",
                "extra_cmds",
            }
        }
        out.append(
            {
                "lib": _to_str(raw.get("lib")),
                "copy_skeleton": raw.get("copy_skeleton") if isinstance(raw.get("copy_skeleton"), bool) else None,
                "packages": _to_string_list(raw.get("packages")),
                "extra_coq_proj_args": _to_string_list(raw.get("extra_coq_proj_args")),
                "extra_coq_makefile_args": _to_string_list(raw.get("extra_coq_makefile_args")),
                "extra_cmds": _to_string_list(raw.get("extra_cmds")),
                "extra_json": _json_compact(extra) if extra else None,
            }
        )
    return out


def _location_type() -> pa.DataType:
    return pa.struct(
        [
            ("range_start_line", pa.int64()),
            ("range_start_character", pa.int64()),
            ("range_end_line", pa.int64()),
            ("range_end_character", pa.int64()),
        ]
    )


def _symbol_ref_type() -> pa.DataType:
    return pa.struct(
        [
            ("name", pa.string()),
            ("logical_path", pa.string()),
            ("physical_path", pa.string()),
            ("locations", pa.list_(_location_type())),
        ]
    )


def _goal_hyp_type() -> pa.DataType:
    return pa.struct(
        [
            ("names", pa.list_(pa.string())),
            ("ty", pa.string()),
        ]
    )


def _goal_type() -> pa.DataType:
    return pa.struct(
        [
            ("evar", pa.int64()),
            ("name", pa.string()),
            ("hyps", pa.list_(_goal_hyp_type())),
            ("ty", pa.string()),
        ]
    )


def _goal_state_type() -> pa.DataType:
    return pa.struct(
        [
            ("goals", pa.list_(_goal_type())),
            ("stack_json", pa.list_(pa.string())),
            ("bullet", pa.string()),
            ("shelf_json", pa.list_(pa.string())),
            ("given_up_json", pa.list_(pa.string())),
        ]
    )


def _target_type() -> pa.DataType:
    return pa.struct(
        [
            ("lib", pa.string()),
            ("copy_skeleton", pa.bool_()),
            ("packages", pa.list_(pa.string())),
            ("extra_coq_proj_args", pa.list_(pa.string())),
            ("extra_coq_makefile_args", pa.list_(pa.string())),
            ("extra_cmds", pa.list_(pa.string())),
            ("extra_json", pa.string()),
        ]
    )


# ---------------- UID matching helpers (internal only) ----------------

def _span_offsets(element: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
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


def _extract_target_elements(toc: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    target_elements: List[Dict[str, Any]] = []
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

            element = dict(entry)
            element_data = element.get("data")
            if not isinstance(element_data, Mapping):
                element_data = {}
            element_data = dict(element_data)
            element_data["fqn"] = fqn
            element_data["history"] = ".".join(history)
            element["data"] = element_data
            target_elements.append(element)
        elif kind in PROOF_KINDS:
            pass

    return target_elements


def _make_safe_uid(raw_uid: str, used: set[str]) -> str:
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


def _assign_uids(target_elements: List[Dict[str, Any]]) -> None:
    idx_notation = sum(1 for e in target_elements if e.get("kind") in NOTATION_KINDS)
    idx_uid: DefaultDict[str, int] = defaultdict(int)
    used_uids: set[str] = set()

    sorted_targets = sorted(
        target_elements,
        key=lambda e: _span_offsets(e)[0] if _span_offsets(e) else -1,
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

        uid = _make_safe_uid(raw_uid=raw_uid, used=used_uids)
        used_uids.add(uid)

        idx_uid[fqn] += 1
        data["uid"] = uid
        element["data"] = data


def _assign_legacy_uids(target_elements: List[Dict[str, Any]]) -> None:
    idx_notation = sum(1 for e in target_elements if e.get("kind") in NOTATION_KINDS)
    idx_uid: DefaultDict[str, int] = defaultdict(int)

    sorted_targets = sorted(
        target_elements,
        key=lambda e: _span_offsets(e)[0] if _span_offsets(e) else -1,
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
            uid = f"NOTATION_{idx_notation}"
            idx_notation -= 1
        elif fqn in idx_uid:
            uid = f"{fqn}_duplicata_{idx_uid[fqn]}"
        else:
            uid = fqn

        idx_uid[fqn] += 1
        data["uid"] = uid
        element["data"] = data


def _normalize_uid_sequence_for_current_pipeline(raw_uids: Sequence[str]) -> List[str]:
    out: List[str] = []
    used: set[str] = set()
    for raw_uid in raw_uids:
        if not isinstance(raw_uid, str):
            continue
        normalized = _make_safe_uid(raw_uid=raw_uid, used=used)
        used.add(normalized)
        out.append(normalized)
    return out


def _target_fingerprint(element: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    kind = element.get("kind")
    if not isinstance(kind, str):
        return None

    span = element.get("span")
    if not isinstance(span, Mapping):
        return None
    bp = span.get("bp")
    ep = span.get("ep")
    if not isinstance(bp, int) or not isinstance(ep, int):
        return None

    name = element.get("name") if isinstance(element.get("name"), str) else None
    data = element.get("data")
    if not isinstance(data, Mapping):
        data = {}
    fqn = data.get("fqn") if isinstance(data.get("fqn"), str) else None
    history = data.get("history") if isinstance(data.get("history"), str) else None

    return (kind, name, bp, ep, fqn, history)


def _target_fingerprint_candidates(element: Mapping[str, Any]) -> List[Tuple[Any, ...]]:
    full = _target_fingerprint(element)
    if full is None:
        return []
    kind, name, bp, ep, fqn, history = full
    return [
        ("full", kind, name, bp, ep, fqn, history),
        ("fqn", kind, name, bp, ep, fqn),
        ("span", kind, name, bp, ep),
    ]


def _build_uid_lookup_from_target_elements(
    target_elements: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[Any, ...], Optional[str]]:
    out: Dict[Tuple[Any, ...], Optional[str]] = {}
    for element in target_elements:
        if not isinstance(element, Mapping):
            continue
        data = element.get("data")
        uid = data.get("uid") if isinstance(data, Mapping) else None
        if not isinstance(uid, str):
            continue
        for fp in _target_fingerprint_candidates(element):
            if fp not in out:
                out[fp] = uid
            elif out[fp] != uid:
                out[fp] = None
    return out


def _resolve_uid_from_lookup(
    element: Mapping[str, Any],
    uid_lookup: Mapping[Tuple[Any, ...], Optional[str]],
) -> Optional[str]:
    for fp in _target_fingerprint_candidates(element):
        uid = uid_lookup.get(fp)
        if isinstance(uid, str):
            return uid
    return None


class BufferedParquetWriter:
    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        batch_size: int = 20_000,
        compression: str = "zstd",
    ) -> None:
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self.compression = compression
        self._buffer: List[Dict[str, Any]] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self.rows_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: Dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, self.schema, compression=self.compression)
        self._writer.write_table(table)
        self.rows_written += table.num_rows
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            return
        columns = [pa.array([], type=field.type) for field in self.schema]
        table = pa.Table.from_arrays(columns, names=self.schema.names)
        pq.write_table(table, self.path, compression=self.compression)


def _schemas(with_lossless_blobs: bool) -> Dict[str, pa.Schema]:
    schemas: Dict[str, pa.Schema] = {
        "sources": pa.schema(
            [
                ("env_id", pa.string()),
                ("source_id", pa.string()),
                ("source_idx", pa.int64()),
                ("source_entry_key", pa.string()),
                ("library", pa.string()),
                ("source_path", pa.string()),
                ("source_path_canonical", pa.string()),
                ("source_rel_no_ext", pa.string()),
                ("logical_path", pa.string()),
                ("content", pa.string()),
                ("content_sha256", pa.string()),
                ("need_root", pa.bool_()),
            ]
        ),
        "toc_nodes": pa.schema(
            [
                ("preorder_index", pa.int64()),
                ("sibling_index", pa.int64()),
                ("depth", pa.int64()),
                ("kind", pa.string()),
                ("name", pa.string()),
                ("span_bp", pa.int64()),
                ("span_ep", pa.int64()),
                ("docstring", pa.string()),
                ("data_uid", pa.string()),
                ("data_extra_json", pa.string()),
            ]
        ),
        "proofs": pa.schema(
            [
                ("env_id", pa.string()),
                ("proof_uid", pa.string()),
                ("source_id", pa.string()),
                ("proof_index", pa.int64()),
                ("proof_id", pa.int64()),
                ("name", pa.string()),
                ("statement", pa.string()),
                ("start_line", pa.int64()),
                ("start_character", pa.int64()),
                ("end_line", pa.int64()),
                ("end_character", pa.int64()),
                ("axioms", pa.list_(_symbol_ref_type())),
                ("initial_goals", _goal_state_type()),
                ("statement_notations", pa.list_(_symbol_ref_type())),
                ("num_steps", pa.int64()),
            ]
        ),
        "proof_steps": pa.schema(
            [
                ("env_id", pa.string()),
                ("step_uid", pa.string()),
                ("proof_uid", pa.string()),
                ("source_id", pa.string()),
                ("step_order", pa.int64()),
                ("step_index", pa.int64()),
                ("raw", pa.string()),
                ("tactic_tags", pa.list_(pa.string())),
                ("notations", pa.list_(_symbol_ref_type())),
                ("deps", pa.list_(_symbol_ref_type())),
                ("range_start_line", pa.int64()),
                ("range_start_character", pa.int64()),
                ("range_end_line", pa.int64()),
                ("range_end_character", pa.int64()),
                ("goals_after", _goal_state_type()),
                ("num_deps", pa.int64()),
            ]
        ),
        "step_deps": pa.schema(
            [
                ("env_id", pa.string()),
                ("dep_uid", pa.string()),
                ("step_uid", pa.string()),
                ("proof_uid", pa.string()),
                ("source_id", pa.string()),
                ("dep_order", pa.int64()),
                ("name", pa.string()),
                ("logical_path", pa.string()),
                ("physical_path", pa.string()),
                ("locations", pa.list_(_location_type())),
            ]
        ),
        "proof_axioms": pa.schema(
            [
                ("env_id", pa.string()),
                ("axiom_uid", pa.string()),
                ("proof_uid", pa.string()),
                ("source_id", pa.string()),
                ("axiom_order", pa.int64()),
                ("name", pa.string()),
                ("logical_path", pa.string()),
                ("physical_path", pa.string()),
                ("locations", pa.list_(_location_type())),
            ]
        ),
        "env_toc": pa.schema(
            [
                ("env_id", pa.string()),
                ("node_id", pa.string()),
                ("parent_id", pa.string()),
                ("preorder_index", pa.int64()),
                ("sibling_index", pa.int64()),
                ("depth", pa.int64()),
                ("node_type", pa.string()),
                ("name", pa.string()),
                ("path", pa.string()),
                ("entry_key", pa.string()),
                ("annotation_count", pa.int64()),
                ("one_liner", pa.string()),
                ("num_children", pa.int64()),
                ("children_ids", pa.list_(pa.string())),
                ("source_id", pa.string()),
                ("source_path_canonical", pa.string()),
                ("has_source_match", pa.bool_()),
            ]
        ),
        "env_metadata": pa.schema(
            [
                ("env_id", pa.string()),
                ("config_filename", pa.string()),
                ("export_filename", pa.string()),
                ("docstrings_filename", pa.string()),
                ("env_toc_filename", pa.string()),
                ("config_name", pa.string()),
                ("version", pa.string()),
                ("tag", pa.string()),
                ("base_image", pa.string()),
                ("user", pa.string()),
                ("opam_env_path", pa.string()),
                ("packages", pa.list_(pa.string())),
                ("dependencies", pa.list_(pa.string())),
                ("pins", pa.list_(pa.string())),
                ("targets", pa.list_(_target_type())),
                ("generated_at_utc", pa.string()),
                ("schema_version", pa.int64()),
                ("source_rows", pa.int64()),
                ("toc_rows", pa.int64()),
                ("target_toc_rows", pa.int64()),
                ("annotated_toc_rows", pa.int64()),
                ("docstring_coverage", pa.float64()),
                ("proof_rows", pa.int64()),
                ("proof_step_rows", pa.int64()),
                ("dep_rows", pa.int64()),
                ("axiom_rows", pa.int64()),
                ("env_toc_rows", pa.int64()),
            ]
        ),
    }
    if with_lossless_blobs:
        schemas["source_blobs"] = pa.schema(
            [
                ("env_id", pa.string()),
                ("source_id", pa.string()),
                ("source_idx", pa.int64()),
                ("entry_json", pa.string()),
            ]
        )
    return schemas


@dataclass
class EnvStats:
    env_id: str
    source_rows: int = 0
    toc_rows: int = 0
    target_toc_rows: int = 0
    annotated_toc_rows: int = 0
    proof_rows: int = 0
    proof_step_rows: int = 0
    dep_rows: int = 0
    axiom_rows: int = 0
    env_toc_rows: int = 0


@dataclass
class TocBuildStats:
    total_rows: int = 0
    target_rows: int = 0
    annotated_rows: int = 0


@dataclass
class AnnotationBundle:
    annotations: Dict[str, str]
    cache_field: Dict[str, str]
    uid_lookup: Dict[Tuple[Any, ...], Optional[str]]


def _as_string_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not _is_non_empty_string(v):
            continue
        out[k] = v
    return out


def _merge_string_maps(base: Dict[str, str], incoming: Mapping[str, str]) -> None:
    for key, value in incoming.items():
        if key not in base:
            base[key] = value
            continue
        if base[key] == value:
            continue
        chosen = _prefer_longer_string(base[key], value)
        if chosen is not None:
            base[key] = chosen


def _merge_uid_lookups(
    base: Dict[Tuple[Any, ...], Optional[str]],
    incoming: Mapping[Tuple[Any, ...], Optional[str]],
) -> None:
    for fp, uid in incoming.items():
        if fp not in base:
            base[fp] = uid
            continue
        if base[fp] == uid:
            continue
        base[fp] = None


def _build_uid_lookup_from_toc(toc: Any) -> Dict[Tuple[Any, ...], Optional[str]]:
    if not isinstance(toc, Sequence) or isinstance(toc, (str, bytes, bytearray)):
        return {}
    raw_items = [x for x in toc if isinstance(x, Mapping)]
    targets = _extract_target_elements(raw_items)
    if not all(
        isinstance((target.get("data") if isinstance(target.get("data"), Mapping) else {}).get("uid"), str)
        for target in targets
    ):
        _assign_uids(targets)
    return _build_uid_lookup_from_target_elements(targets)


def _merge_annotation_bundle(existing: AnnotationBundle, incoming: AnnotationBundle) -> None:
    _merge_string_maps(existing.annotations, incoming.annotations)
    _merge_string_maps(existing.cache_field, incoming.cache_field)
    _merge_uid_lookups(existing.uid_lookup, incoming.uid_lookup)


def _load_annotation_indexes(
    merged_env_file: Path,
) -> Tuple[
    Dict[Tuple[str, str, str], AnnotationBundle],
    Dict[Tuple[str, str], AnnotationBundle],
]:
    by_sig: Dict[Tuple[str, str, str], AnnotationBundle] = {}
    by_path: Dict[Tuple[str, str], AnnotationBundle] = {}

    with merged_env_file.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{merged_env_file}:{line_idx}: invalid JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                continue

            library = row.get("library")
            source = row.get("source")
            if not isinstance(library, str) or not isinstance(source, Mapping):
                continue
            source_path = source.get("path")
            source_content = source.get("content")
            if not isinstance(source_path, str) or not isinstance(source_content, str):
                continue

            rel_no_ext = _source_relative_without_ext(source_path, library)
            source_sha = _sha256_text(source_content)
            sig_key = (library, rel_no_ext, source_sha)
            path_key = (library, rel_no_ext)

            bundle = AnnotationBundle(
                annotations=_as_string_map(row.get("annotations")),
                cache_field=_as_string_map(row.get("cache_field")),
                uid_lookup=_build_uid_lookup_from_toc(row.get("toc")),
            )

            if sig_key in by_sig:
                _merge_annotation_bundle(by_sig[sig_key], bundle)
            else:
                by_sig[sig_key] = bundle

            if path_key in by_path:
                _merge_annotation_bundle(by_path[path_key], bundle)
            else:
                by_path[path_key] = AnnotationBundle(
                    annotations=dict(bundle.annotations),
                    cache_field=dict(bundle.cache_field),
                    uid_lookup=dict(bundle.uid_lookup),
                )

    return by_sig, by_path


def _pick_annotation_bundle(
    by_sig: Mapping[Tuple[str, str, str], AnnotationBundle],
    by_path: Mapping[Tuple[str, str], AnnotationBundle],
    library: str,
    rel_no_ext: str,
    source_sha: str,
) -> Optional[AnnotationBundle]:
    sig_key = (library, rel_no_ext, source_sha)
    if sig_key in by_sig:
        return by_sig[sig_key]
    path_key = (library, rel_no_ext)
    if path_key in by_path:
        return by_path[path_key]
    return None


def _build_target_uid_plan(
    source_toc: Any,
    uid_lookup: Mapping[Tuple[Any, ...], Optional[str]],
) -> Dict[Tuple[Tuple[Any, ...], int], Optional[str]]:
    if not isinstance(source_toc, Sequence) or isinstance(source_toc, (str, bytes, bytearray)):
        return {}

    targets = _extract_target_elements([x for x in source_toc if isinstance(x, Mapping)])
    _assign_legacy_uids(targets)

    legacy_raw_uids: List[str] = []
    for target in targets:
        data = target.get("data")
        uid = data.get("uid") if isinstance(data, Mapping) else None
        if isinstance(uid, str):
            legacy_raw_uids.append(uid)
        else:
            legacy_raw_uids.append("uid")
    legacy_uids = _normalize_uid_sequence_for_current_pipeline(legacy_raw_uids)

    plan: Dict[Tuple[Tuple[Any, ...], int], Optional[str]] = {}
    occ: DefaultDict[Tuple[Any, ...], int] = defaultdict(int)

    for idx, target in enumerate(targets):
        fp = _target_fingerprint(target)
        if fp is None:
            continue

        n = occ[fp]
        occ[fp] += 1

        data = target.get("data")
        explicit_uid = data.get("uid") if isinstance(data, Mapping) else None
        if isinstance(explicit_uid, str):
            plan[(fp, n)] = explicit_uid
            continue

        resolved_uid = _resolve_uid_from_lookup(target, uid_lookup)
        if isinstance(resolved_uid, str):
            plan[(fp, n)] = resolved_uid
            continue

        fallback_uid = legacy_uids[idx] if idx < len(legacy_uids) else None
        plan[(fp, n)] = fallback_uid if isinstance(fallback_uid, str) else None

    return plan


def _pick_docstring(uid: Optional[str], annotations: Mapping[str, str], cache_field: Mapping[str, str]) -> Optional[str]:
    if not isinstance(uid, str):
        return None
    from_annotations = annotations.get(uid)
    if _is_non_empty_string(from_annotations):
        return from_annotations
    from_cache = cache_field.get(uid)
    if _is_non_empty_string(from_cache):
        return from_cache
    return None


def _iter_toc_rows(
    env_id: str,
    source_id: str,
    nodes: Any,
    annotation_bundle: Optional[AnnotationBundle],
    stats: TocBuildStats,
) -> Iterator[Dict[str, Any]]:
    preorder = 0
    fp_seen: DefaultDict[Tuple[Any, ...], int] = defaultdict(int)
    namespaces_stack: List[Tuple[str, str]] = []

    annotations: Mapping[str, str] = annotation_bundle.annotations if annotation_bundle else {}
    cache_field: Mapping[str, str] = annotation_bundle.cache_field if annotation_bundle else {}
    uid_lookup: Mapping[Tuple[Any, ...], Optional[str]] = annotation_bundle.uid_lookup if annotation_bundle else {}
    uid_plan = _build_target_uid_plan(nodes, uid_lookup)

    def walk(current_nodes: Any, parent_toc_id: Optional[str], depth: int) -> Iterator[Dict[str, Any]]:
        nonlocal preorder
        if not isinstance(current_nodes, list):
            return

        for sibling_idx, node in enumerate(current_nodes):
            if not isinstance(node, dict):
                continue

            toc_id = f"{source_id}:toc:{preorder}"
            preorder_idx = preorder
            preorder += 1

            kind = _to_str(node.get("kind"))
            name = _to_str(node.get("name"))
            data = node.get("data") if isinstance(node.get("data"), Mapping) else {}
            span = _parse_span(node.get("span"))
            rng = _parse_range(node.get("range"))

            if kind == "begin_section":
                if isinstance(name, str) and name:
                    namespaces_stack.append(("SECTION", name))
            elif kind in {"declare_module_type", "define_module"}:
                is_alias = bool(data.get("is_alias", False)) if isinstance(data, Mapping) else False
                if not is_alias and isinstance(name, str) and name:
                    namespaces_stack.append(("MODULE", name))
            elif kind == "end_segment":
                if namespaces_stack:
                    namespaces_stack.pop()

            docstring: Optional[str] = None
            data_uid, _data_fqn, _data_history, _data_is_alias, data_extra_json = _extract_toc_data_fields(node.get("data"))
            candidate_target = bool(kind in TARGET_KINDS and isinstance(name, str) and name)
            if candidate_target:
                stats.target_rows += 1
                explicit_doc = node.get("docstring")
                if _is_non_empty_string(explicit_doc):
                    docstring = explicit_doc

                explicit_uid = data.get("uid") if isinstance(data, Mapping) else None
                resolved_uid: Optional[str] = explicit_uid if isinstance(explicit_uid, str) else None

                stack_modules = [n for scope, n in namespaces_stack if scope == "MODULE"]
                history_parts = [n for _, n in namespaces_stack]
                fqn = ".".join(stack_modules + [name]) if stack_modules else name
                history = ".".join(history_parts)

                bp = span.get("bp")
                ep = span.get("ep")
                fp: Optional[Tuple[Any, ...]] = None
                if isinstance(bp, int) and isinstance(ep, int):
                    fp = (kind, name, bp, ep, fqn, history)

                if fp is not None:
                    seen_count = fp_seen[fp]
                    fp_seen[fp] += 1
                    if resolved_uid is None and (fp, seen_count) in uid_plan:
                        uid = uid_plan[(fp, seen_count)]
                        resolved_uid = uid if isinstance(uid, str) else None

                if docstring is None:
                    docstring = _pick_docstring(resolved_uid, annotations, cache_field)
                if docstring is not None:
                    stats.annotated_rows += 1

            row = {
                "preorder_index": preorder_idx,
                "sibling_index": sibling_idx,
                "depth": depth,
                "kind": kind,
                "name": name,
                "span_bp": span["bp"],
                "span_ep": span["ep"],
                "docstring": docstring,
                "data_uid": data_uid,
                "data_extra_json": data_extra_json,
            }
            stats.total_rows += 1
            yield row
            yield from walk(node.get("members"), toc_id, depth + 1)

    yield from walk(nodes, None, 0)


def _iter_env_toc_rows(
    env_id: str,
    toc_doc: Mapping[str, Any],
    source_by_entry_key: Mapping[str, Tuple[str, str]],
) -> Iterator[Dict[str, Any]]:
    raw_nodes = toc_doc.get("nodes")
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes, bytearray)):
        return

    node_order: List[str] = []
    node_map: Dict[str, Dict[str, Any]] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping):
            continue
        node_id = raw_node.get("id")
        if not isinstance(node_id, str):
            continue
        if node_id in node_map:
            continue
        node_order.append(node_id)
        node_map[node_id] = dict(raw_node)

    if not node_map:
        return

    node_to_entry_key: Dict[str, str] = {}
    raw_index = toc_doc.get("file_index")
    if isinstance(raw_index, Mapping):
        for entry_key, node_id in raw_index.items():
            if not isinstance(entry_key, str) or not isinstance(node_id, str):
                continue
            if node_id not in node_to_entry_key:
                node_to_entry_key[node_id] = entry_key

    root_id = toc_doc.get("root_id") if isinstance(toc_doc.get("root_id"), str) else None
    preorder = 0
    visited: set[str] = set()

    def emit(node_id: str, depth: int, sibling_index: int, parent_hint: Optional[str]) -> Iterator[Dict[str, Any]]:
        nonlocal preorder
        if node_id in visited:
            return
        node = node_map.get(node_id)
        if node is None:
            return

        visited.add(node_id)
        children = _normalize_children(node.get("children_ids"))

        parent_id_value = _to_str(node.get("parent_id"))
        if parent_hint is not None:
            parent_id_value = parent_hint

        entry_key = _to_str(node.get("entry_key"))
        if entry_key is None:
            entry_key = node_to_entry_key.get(node_id)

        source = source_by_entry_key.get(entry_key) if isinstance(entry_key, str) else None
        source_id = source[0] if source is not None else None
        source_path_canonical = source[1] if source is not None else None

        row = {
            "env_id": env_id,
            "node_id": node_id,
            "parent_id": parent_id_value,
            "preorder_index": preorder,
            "sibling_index": sibling_index,
            "depth": depth,
            "node_type": _to_str(node.get("type")),
            "name": _to_str(node.get("name")),
            "path": _to_str(node.get("path")),
            "entry_key": entry_key,
            "annotation_count": _to_int(node.get("annotation_count")),
            "one_liner": _to_str(node.get("one_liner")),
            "num_children": len(children),
            "children_ids": children,
            "source_id": source_id,
            "source_path_canonical": source_path_canonical,
            "has_source_match": source is not None,
        }
        preorder += 1
        yield row

        for idx, child_id in enumerate(children):
            yield from emit(child_id, depth + 1, idx, node_id)

    start_nodes: List[str] = []
    if isinstance(root_id, str) and root_id in node_map:
        start_nodes.append(root_id)
    else:
        for node_id in node_order:
            parent_id = _to_str(node_map[node_id].get("parent_id"))
            if parent_id not in node_map:
                start_nodes.append(node_id)

    for idx, node_id in enumerate(start_nodes):
        yield from emit(node_id, depth=0, sibling_index=idx, parent_hint=None)

    for node_id in node_order:
        if node_id in visited:
            continue
        yield from emit(node_id, depth=0, sibling_index=0, parent_hint=None)


def _table_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _docstring_coverage(stats: EnvStats) -> float:
    if stats.target_toc_rows <= 0:
        return 1.0
    return stats.annotated_toc_rows / stats.target_toc_rows


def _write_dataset_card(env_dir: Path, env_id: str, tables: Sequence[str], stats: EnvStats) -> None:
    config_lines: List[str] = []
    for table in tables:
        config_lines.extend(
            [
                "- config_name: " + table,
                "  data_files:",
                "  - split: train",
                f"    path: {table}.parquet",
            ]
        )

    coverage = _docstring_coverage(stats)

    readme = (
        "---\n"
        "configs:\n"
        + "\n".join(config_lines)
        + "\n"
        "---\n\n"
        f"# Pile-of-Rocq `{env_id}`\n\n"
        "Normalized extraction dump enriched with source-level `docstring` and env-level TOC.\n\n"
        "## Tables\n"
        + "\n".join(f"- `{table}`" for table in tables)
        + "\n\n"
        "## Load examples\n"
        "```python\n"
        "from datasets import load_dataset\n\n"
        f"toc = load_dataset('YOUR_ORG/{env_id}', 'toc_nodes', split='train')\n"
        f"env_toc = load_dataset('YOUR_ORG/{env_id}', 'env_toc', split='train')\n"
        "print(toc.column_names)\n"
        "print(env_toc.column_names)\n"
        "```\n\n"
        "## Basic counts\n"
        f"- sources: {stats.source_rows}\n"
        f"- source toc nodes: {stats.toc_rows}\n"
        f"- source toc targets: {stats.target_toc_rows}\n"
        f"- source toc targets with docstring: {stats.annotated_toc_rows}\n"
        f"- source toc docstring coverage: {coverage:.4f}\n"
        f"- proofs: {stats.proof_rows}\n"
        f"- proof steps: {stats.proof_step_rows}\n"
        f"- step deps: {stats.dep_rows}\n"
        f"- proof axioms: {stats.axiom_rows}\n"
        f"- env toc nodes: {stats.env_toc_rows}\n"
    )
    (env_dir / "README.md").write_text(readme, encoding="utf-8")


def _write_root_dataset_card(
    output_root: Path,
    repo_id: str,
    env_entries: Sequence[Dict[str, Any]],
) -> None:
    config_lines: List[str] = []
    sorted_entries = sorted(env_entries, key=lambda x: x["env_id"])
    for entry in sorted_entries:
        env_id = entry["env_id"]
        for table in entry["tables"]:
            config_name = f"{env_id}-{table}"
            config_lines.extend(
                [
                    "- config_name: " + config_name,
                    "  data_files:",
                    "  - split: train",
                    f"    path: {env_id}/{table}.parquet",
                ]
            )

    env_count = len(sorted_entries)
    readme = (
        "---\n"
        "configs:\n"
        + "\n".join(config_lines)
        + "\n"
        "---\n\n"
        f"# {repo_id}\n\n"
        "Pile-of-Rocq exported as normalized parquet tables with `docstring` + `env_toc`.\n\n"
        "Each config is `<env>-<table>` and loads one parquet table for one env.\n\n"
        "## Load examples\n"
        "```python\n"
        "from datasets import load_dataset\n\n"
        f"toc = load_dataset('{repo_id}', 'coq-actuary-toc_nodes', split='train')\n"
        f"steps = load_dataset('{repo_id}', 'coq-actuary-proof_steps', split='train')\n"
        "print(len(toc), len(steps))\n"
        "```\n\n"
        f"## Environments in this export\n- count: {env_count}\n"
        + "\n".join(f"- `{entry['env_id']}`" for entry in sorted_entries)
        + "\n"
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def _expected_tables(with_lossless_blobs: bool) -> List[str]:
    tables = list(BASE_TABLES_ORDER)
    if with_lossless_blobs:
        tables.append("source_blobs")
    return tables


def _env_export_complete(env_dir: Path, with_lossless_blobs: bool) -> bool:
    if not env_dir.exists() or not env_dir.is_dir():
        return False
    if not (env_dir / "manifest.json").exists():
        return False
    for table in _expected_tables(with_lossless_blobs):
        if not (env_dir / f"{table}.parquet").exists():
            return False
    return True


def _summary_from_existing_env(env_dir: Path, env_id: str, with_lossless_blobs: bool) -> Tuple[EnvStats, List[str]]:
    manifest_path = env_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stats_json = manifest.get("stats")
            if isinstance(stats_json, Mapping):
                stats = EnvStats(
                    env_id=env_id,
                    source_rows=_to_int(stats_json.get("source_rows")) or 0,
                    toc_rows=_to_int(stats_json.get("toc_rows")) or 0,
                    target_toc_rows=_to_int(stats_json.get("target_toc_rows")) or 0,
                    annotated_toc_rows=_to_int(stats_json.get("annotated_toc_rows")) or 0,
                    proof_rows=_to_int(stats_json.get("proof_rows")) or 0,
                    proof_step_rows=_to_int(stats_json.get("proof_step_rows")) or 0,
                    dep_rows=_to_int(stats_json.get("dep_rows")) or 0,
                    axiom_rows=_to_int(stats_json.get("axiom_rows")) or 0,
                    env_toc_rows=_to_int(stats_json.get("env_toc_rows")) or 0,
                )
                return stats, _expected_tables(with_lossless_blobs)
        except Exception:
            pass

    stats = EnvStats(
        env_id=env_id,
        source_rows=_table_row_count(env_dir / "sources.parquet"),
        toc_rows=_table_row_count(env_dir / "toc_nodes.parquet"),
        proof_rows=_table_row_count(env_dir / "proofs.parquet"),
        proof_step_rows=_table_row_count(env_dir / "proof_steps.parquet"),
        dep_rows=_table_row_count(env_dir / "step_deps.parquet"),
        axiom_rows=_table_row_count(env_dir / "proof_axioms.parquet"),
        env_toc_rows=_table_row_count(env_dir / "env_toc.parquet"),
    )
    return stats, _expected_tables(with_lossless_blobs)


def _folder_stats(folder_path: Path) -> Tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for path in folder_path.rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return file_count, total_bytes


def _pick_upload_method(upload_method: str, file_count: int, total_bytes: int) -> str:
    if upload_method != "auto":
        return upload_method
    if file_count >= 200 or total_bytes >= 2 * 1024**3:
        return "large-folder"
    return "upload-folder"


def _push_folder_to_hub(
    folder_path: Path,
    repo_id: str,
    private: bool,
    token: Optional[str],
    commit_message: str,
    upload_method: str,
    upload_num_workers: Optional[int],
    upload_retries: int,
    retry_backoff_sec: float,
) -> None:
    if HfApi is None:
        raise RuntimeError("huggingface_hub is not installed but --push-to-hub was requested.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    file_count, total_bytes = _folder_stats(folder_path)
    chosen_method = _pick_upload_method(upload_method, file_count, total_bytes)
    print(
        f"[upload] repo={repo_id} files={file_count} size_gb={total_bytes / (1024**3):.2f} "
        f"method={chosen_method}"
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, upload_retries + 1):
        try:
            if chosen_method == "large-folder":
                kwargs: Dict[str, Any] = {}
                if upload_num_workers is not None:
                    kwargs["num_workers"] = upload_num_workers
                api.upload_large_folder(
                    repo_id=repo_id,
                    folder_path=str(folder_path),
                    repo_type="dataset",
                    revision=None,
                    private=private,
                    **kwargs,
                )
            else:
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=str(folder_path),
                    path_in_repo=".",
                    commit_message=commit_message,
                )
            return
        except Exception as exc:
            last_error = exc
            if attempt >= upload_retries:
                break
            if chosen_method == "upload-folder":
                chosen_method = "large-folder"
            sleep_sec = retry_backoff_sec * (2 ** (attempt - 1))
            print(
                f"[upload] attempt {attempt}/{upload_retries} failed with {type(exc).__name__}: {exc}. "
                f"Retrying in {sleep_sec:.1f}s (method={chosen_method})."
            )
            time.sleep(sleep_sec)

    assert last_error is not None
    raise last_error


def _build_repo_id(namespace: str, repo_name: str) -> str:
    return f"{namespace}/{repo_name}"


def _build_env_repo_id(namespace: str, prefix: str, env_id: str) -> str:
    return f"{namespace}/{prefix}{env_id}"


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
    if not isinstance(content, dict):
        raise ValueError(f"Config is not a mapping: {config_path}")
    return content


def _load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _convert_env(
    env_id: str,
    export_file: Path,
    config_file: Path,
    docstrings_file: Path,
    env_toc_file: Path,
    output_root: Path,
    batch_size: int,
    compression: str,
    with_lossless_blobs: bool,
    strict_docstrings: bool,
    strict_env_toc: bool,
) -> Tuple[Path, EnvStats, List[str]]:
    env_dir = output_root / env_id
    env_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_file)

    ann_by_sig: Dict[Tuple[str, str, str], AnnotationBundle] = {}
    ann_by_path: Dict[Tuple[str, str], AnnotationBundle] = {}
    if docstrings_file.exists():
        ann_by_sig, ann_by_path = _load_annotation_indexes(docstrings_file)
    elif strict_docstrings:
        raise FileNotFoundError(f"[{env_id}] missing merged docstrings file: {docstrings_file}")

    env_toc_doc = _load_optional_json(env_toc_file)
    if env_toc_doc is None and strict_env_toc:
        raise FileNotFoundError(f"[{env_id}] missing env TOC file: {env_toc_file}")

    schemas = _schemas(with_lossless_blobs)
    writers = {
        name: BufferedParquetWriter(
            path=env_dir / f"{name}.parquet",
            schema=schema,
            batch_size=batch_size,
            compression=compression,
        )
        for name, schema in schemas.items()
    }

    stats = EnvStats(env_id=env_id)
    generated_at = datetime.now(timezone.utc).isoformat()
    source_by_entry_key: Dict[str, Tuple[str, str]] = {}

    try:
        for source_idx, entry, raw_line in _iter_jsonl(export_file):
            if not isinstance(entry, dict):
                continue

            source_obj = entry.get("source") or {}
            if not isinstance(source_obj, dict):
                source_obj = {}

            library = _to_str(entry.get("library")) or "UNKNOWN"
            source_path = _to_str(source_obj.get("path")) or ""
            source_content = _to_str(source_obj.get("content")) or ""
            logical_path = _to_str(source_obj.get("logical_path"))
            need_root = bool(entry.get("need_root"))

            source_id, content_sha, canonical_path = _source_identity(library, source_path, source_content)
            source_rel_no_ext = _source_relative_without_ext(source_path, library)
            source_entry_key = _entry_key_from_source(library, source_path, source_content)
            source_by_entry_key.setdefault(source_entry_key, (source_id, canonical_path))

            toc_nodes = entry.get("toc") if isinstance(entry.get("toc"), list) else []
            proofs_obj = entry.get("proofs") if isinstance(entry.get("proofs"), dict) else {}
            proofs = proofs_obj.get("proofs") if isinstance(proofs_obj.get("proofs"), list) else []

            ann_bundle = _pick_annotation_bundle(
                ann_by_sig,
                ann_by_path,
                library,
                source_rel_no_ext,
                content_sha,
            )
            if strict_docstrings and ann_bundle is None:
                raise RuntimeError(
                    f"[{env_id}] source has no docstring match: library={library} rel={source_rel_no_ext}"
                )

            toc_build_stats = TocBuildStats()
            for toc_row in _iter_toc_rows(env_id, source_id, toc_nodes, ann_bundle, toc_build_stats):
                writers["toc_nodes"].write(toc_row)

            stats.toc_rows += toc_build_stats.total_rows
            stats.target_toc_rows += toc_build_stats.target_rows
            stats.annotated_toc_rows += toc_build_stats.annotated_rows

            writers["sources"].write(
                {
                    "env_id": env_id,
                    "source_id": source_id,
                    "source_idx": source_idx,
                    "source_entry_key": source_entry_key,
                    "library": library,
                    "source_path": source_path,
                    "source_path_canonical": canonical_path,
                    "source_rel_no_ext": source_rel_no_ext,
                    "logical_path": logical_path,
                    "content": source_content,
                    "content_sha256": content_sha,
                    "need_root": need_root,
                }
            )
            stats.source_rows += 1

            if with_lossless_blobs:
                writers["source_blobs"].write(
                    {
                        "env_id": env_id,
                        "source_id": source_id,
                        "source_idx": source_idx,
                        "entry_json": raw_line,
                    }
                )

            for proof_index, proof in enumerate(proofs):
                if not isinstance(proof, dict):
                    continue

                proof_id = _to_int(proof.get("proof_id"))
                proof_name = _to_str(proof.get("name")) or ""
                start_range = _parse_range(proof.get("start_range"))
                statement = _to_str(proof.get("statement")) or ""
                proof_uid = _sha256_text(
                    f"{source_id}:proof:{proof_index}:{proof_id}:{proof_name}:{start_range['start_line']}"
                )

                axioms = proof.get("axioms") if isinstance(proof.get("axioms"), list) else []
                steps = proof.get("steps") if isinstance(proof.get("steps"), list) else []

                writers["proofs"].write(
                    {
                        "env_id": env_id,
                        "proof_uid": proof_uid,
                        "source_id": source_id,
                        "proof_index": proof_index,
                        "proof_id": proof_id,
                        "name": proof_name,
                        "statement": statement,
                        "start_line": start_range["start_line"],
                        "start_character": start_range["start_character"],
                        "end_line": start_range["end_line"],
                        "end_character": start_range["end_character"],
                        "axioms": _normalize_symbol_refs(axioms),
                        "initial_goals": _normalize_goal_state(proof.get("initial_goals")),
                        "statement_notations": _normalize_symbol_refs(proof.get("statement_notations")),
                        "num_steps": len(steps),
                    }
                )
                stats.proof_rows += 1

                for axiom_order, axiom in enumerate(axioms):
                    if not isinstance(axiom, dict):
                        continue
                    axiom_uid = _sha256_text(f"{proof_uid}:axiom:{axiom_order}:{axiom.get('name')}")
                    writers["proof_axioms"].write(
                        {
                            "env_id": env_id,
                            "axiom_uid": axiom_uid,
                            "proof_uid": proof_uid,
                            "source_id": source_id,
                            "axiom_order": axiom_order,
                            "name": _to_str(axiom.get("name")),
                            "logical_path": _to_str(axiom.get("logical_path")),
                            "physical_path": _to_str(axiom.get("physical_path")),
                            "locations": _normalize_locations(axiom.get("locations")),
                        }
                    )
                    stats.axiom_rows += 1

                for step_order, step in enumerate(steps):
                    if not isinstance(step, dict):
                        continue
                    step_range = _parse_range(step.get("range"))
                    step_index = _to_int(step.get("index"))
                    step_uid = _sha256_text(
                        f"{proof_uid}:step:{step_order}:{step_index}:{step_range['start_line']}"
                    )
                    deps = step.get("deps") if isinstance(step.get("deps"), list) else []

                    writers["proof_steps"].write(
                        {
                            "env_id": env_id,
                            "step_uid": step_uid,
                            "proof_uid": proof_uid,
                            "source_id": source_id,
                            "step_order": step_order,
                            "step_index": step_index,
                            "raw": _to_str(step.get("raw")),
                            "tactic_tags": _to_string_list(step.get("tactic_tags")),
                            "notations": _normalize_symbol_refs(step.get("notations")),
                            "deps": _normalize_symbol_refs(deps),
                            "range_start_line": step_range["start_line"],
                            "range_start_character": step_range["start_character"],
                            "range_end_line": step_range["end_line"],
                            "range_end_character": step_range["end_character"],
                            "goals_after": _normalize_goal_state(step.get("goals_after")),
                            "num_deps": len(deps),
                        }
                    )
                    stats.proof_step_rows += 1

                    for dep_order, dep in enumerate(deps):
                        if not isinstance(dep, dict):
                            continue
                        dep_uid = _sha256_text(f"{step_uid}:dep:{dep_order}:{dep.get('name')}")
                        writers["step_deps"].write(
                            {
                                "env_id": env_id,
                                "dep_uid": dep_uid,
                                "step_uid": step_uid,
                                "proof_uid": proof_uid,
                                "source_id": source_id,
                                "dep_order": dep_order,
                                "name": _to_str(dep.get("name")),
                                "logical_path": _to_str(dep.get("logical_path")),
                                "physical_path": _to_str(dep.get("physical_path")),
                                "locations": _normalize_locations(dep.get("locations")),
                            }
                        )
                        stats.dep_rows += 1

        if env_toc_doc is not None:
            for row in _iter_env_toc_rows(env_id, env_toc_doc, source_by_entry_key):
                writers["env_toc"].write(row)
                stats.env_toc_rows += 1

        writers["env_metadata"].write(
            {
                "env_id": env_id,
                "config_filename": config_file.name,
                "export_filename": export_file.name,
                "docstrings_filename": docstrings_file.name if docstrings_file.exists() else None,
                "env_toc_filename": env_toc_file.name if env_toc_doc is not None else None,
                "config_name": _to_str(config.get("name")),
                "version": _to_str(config.get("version")),
                "tag": _to_str(config.get("tag")),
                "base_image": _to_str(config.get("base_image")),
                "user": _to_str(config.get("user")),
                "opam_env_path": _to_str(config.get("opam_env_path")),
                "packages": _to_string_list(config.get("packages")),
                "dependencies": _to_string_list(config.get("dependencies")),
                "pins": _to_string_list(config.get("pins")),
                "targets": _normalize_targets(config.get("targets")),
                "generated_at_utc": generated_at,
                "schema_version": SCHEMA_VERSION,
                "source_rows": stats.source_rows,
                "toc_rows": stats.toc_rows,
                "target_toc_rows": stats.target_toc_rows,
                "annotated_toc_rows": stats.annotated_toc_rows,
                "docstring_coverage": _docstring_coverage(stats),
                "proof_rows": stats.proof_rows,
                "proof_step_rows": stats.proof_step_rows,
                "dep_rows": stats.dep_rows,
                "axiom_rows": stats.axiom_rows,
                "env_toc_rows": stats.env_toc_rows,
            }
        )
    finally:
        for writer in writers.values():
            writer.close()

    expected_checks: Dict[str, int] = {
        "sources": stats.source_rows,
        "toc_nodes": stats.toc_rows,
        "proofs": stats.proof_rows,
        "proof_steps": stats.proof_step_rows,
        "step_deps": stats.dep_rows,
        "proof_axioms": stats.axiom_rows,
        "env_toc": stats.env_toc_rows,
        "env_metadata": 1,
    }
    if with_lossless_blobs:
        expected_checks["source_blobs"] = stats.source_rows

    for table_name, expected in expected_checks.items():
        actual = _table_row_count(env_dir / f"{table_name}.parquet")
        if actual != expected:
            raise RuntimeError(
                f"[{env_id}] row-count mismatch for {table_name}: expected={expected} actual={actual}"
            )

    table_names = _expected_tables(with_lossless_blobs)
    _write_dataset_card(env_dir, env_id, table_names, stats)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "env_id": env_id,
        "generated_at_utc": generated_at,
        "input_export_file": str(export_file),
        "input_config_file": str(config_file),
        "input_docstrings_file": str(docstrings_file),
        "input_env_toc_file": str(env_toc_file),
        "tables": table_names,
        "stats": asdict(stats),
    }
    (env_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    return env_dir, stats, table_names


def _discover_envs(export_path: Path, config_path: Path, selected_envs: Optional[Sequence[str]]) -> List[str]:
    export_envs = {file.stem for file in export_path.glob("*.jsonl")}
    config_envs = {file.stem for file in config_path.glob("*.yaml")}

    common = sorted(export_envs & config_envs)
    if selected_envs:
        wanted = {_normalize_env_id(env) for env in selected_envs}
        missing = sorted(wanted - set(common))
        if missing:
            raise FileNotFoundError(
                "Requested env(s) missing either export JSONL or config YAML: " + ", ".join(missing)
            )
        return [env for env in common if env in wanted]
    return common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Pile-of-Rocq dump JSONL to normalized HF-ready parquet with docstring + env_toc."
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("../Pile-of-rocq/export_v2"),
        help="Path to base export JSONL files.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("../Pile-of-rocq/config"),
        help="Path to config YAML files.",
    )
    parser.add_argument(
        "--docstrings-path",
        type=Path,
        default=Path("exported_v3/merged"),
        help="Path to merged docstrings JSONL files (one <env>.jsonl).",
    )
    parser.add_argument(
        "--env-toc-path",
        type=Path,
        default=Path("exported_v3/merged/toc"),
        help="Path to merged env TOC files (one <env>.toc.json).",
    )
    parser.add_argument("--output-path", type=Path, default=Path("hf_export"), help="Output root directory.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment name (file stem, e.g. coq-ssprove). Repeat for multiple envs.",
    )
    parser.add_argument("--batch-size", type=int, default=20_000, help="Rows buffered per parquet write.")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    parser.add_argument(
        "--with-lossless-blobs",
        action="store_true",
        help="Write source_blobs.parquet with original JSON line for exact fidelity.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion for envs already fully exported in --output-path.",
    )
    parser.add_argument(
        "--strict-docstrings",
        action="store_true",
        help="Fail if an env/source has no matching merged docstrings row.",
    )
    parser.add_argument(
        "--strict-env-toc",
        action="store_true",
        help="Fail if merged env TOC file is missing.",
    )

    parser.add_argument("--push-to-hub", action="store_true", help="Upload exported data to HF Hub.")
    parser.add_argument("--hf-namespace", default=None, help="HF namespace/org, required with --push-to-hub.")
    parser.add_argument(
        "--push-layout",
        choices=["single-repo", "per-env-repos"],
        default="single-repo",
        help="Push all envs into one repo, or one repo per env.",
    )
    parser.add_argument(
        "--repo-name",
        default="pile-of-rocq",
        help="Dataset repo name when --push-layout=single-repo.",
    )
    parser.add_argument(
        "--repo-prefix",
        default="pile-of-rocq-",
        help="Repo prefix when --push-layout=per-env-repos. Final repo is <namespace>/<repo-prefix><env>.",
    )
    parser.add_argument("--private", action="store_true", help="Create private dataset repos on HF Hub.")
    parser.add_argument(
        "--upload-method",
        choices=["auto", "upload-folder", "large-folder"],
        default="auto",
        help="Hub upload backend. 'auto' chooses robust mode for large exports.",
    )
    parser.add_argument(
        "--upload-num-workers",
        type=int,
        default=None,
        help="Worker count for large-folder uploads (optional).",
    )
    parser.add_argument(
        "--upload-retries",
        type=int,
        default=3,
        help="Number of upload attempts on transient failures.",
    )
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=8.0,
        help="Initial backoff in seconds before retry; doubles each retry.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token. If omitted, huggingface_hub default authentication is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.push_to_hub and not args.hf_namespace:
        raise ValueError("--hf-namespace is required when --push-to-hub is set.")
    if args.upload_retries < 1:
        raise ValueError("--upload-retries must be >= 1.")
    if args.retry_backoff_sec < 0:
        raise ValueError("--retry-backoff-sec must be >= 0.")

    export_path: Path = args.export_path
    config_path: Path = args.config_path
    docstrings_path: Path = args.docstrings_path
    env_toc_path: Path = args.env_toc_path
    output_path: Path = args.output_path

    if not export_path.exists():
        raise FileNotFoundError(f"Missing export path: {export_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config path: {config_path}")
    if not docstrings_path.exists() and args.strict_docstrings:
        raise FileNotFoundError(f"Missing docstrings path: {docstrings_path}")
    if not env_toc_path.exists() and args.strict_env_toc:
        raise FileNotFoundError(f"Missing env TOC path: {env_toc_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    selected_envs = args.env if args.env else None
    envs = _discover_envs(export_path, config_path, selected_envs)
    if not envs:
        raise RuntimeError("No environments to export.")

    single_repo_id: Optional[str] = None
    if args.push_to_hub and args.push_layout == "single-repo":
        single_repo_id = _build_repo_id(args.hf_namespace, args.repo_name)

    summary: List[Dict[str, Any]] = []
    for env_id in envs:
        export_file = export_path / f"{env_id}.jsonl"
        config_file = config_path / f"{env_id}.yaml"
        docstrings_file = docstrings_path / f"{env_id}.jsonl"
        env_toc_file = env_toc_path / f"{env_id}.toc.json"
        env_dir = output_path / env_id

        if args.skip_existing and _env_export_complete(env_dir, args.with_lossless_blobs):
            stats, table_names = _summary_from_existing_env(env_dir, env_id, args.with_lossless_blobs)
            print(
                f"[{env_id}] skip conversion (already exported): "
                f"sources={stats.source_rows}, targets={stats.target_toc_rows}, "
                f"annotated={stats.annotated_toc_rows}, coverage={_docstring_coverage(stats):.4f}"
            )
        else:
            print(f"[{env_id}] converting...")
            env_dir, stats, table_names = _convert_env(
                env_id=env_id,
                export_file=export_file,
                config_file=config_file,
                docstrings_file=docstrings_file,
                env_toc_file=env_toc_file,
                output_root=output_path,
                batch_size=args.batch_size,
                compression=args.compression,
                with_lossless_blobs=args.with_lossless_blobs,
                strict_docstrings=args.strict_docstrings,
                strict_env_toc=args.strict_env_toc,
            )
            print(
                f"[{env_id}] done: sources={stats.source_rows}, toc={stats.toc_rows}, "
                f"targets={stats.target_toc_rows}, annotated={stats.annotated_toc_rows}, "
                f"coverage={_docstring_coverage(stats):.4f}, proofs={stats.proof_rows}, steps={stats.proof_step_rows}"
            )

        repo_id = None
        if args.push_to_hub and args.push_layout == "per-env-repos":
            repo_id = _build_env_repo_id(args.hf_namespace, args.repo_prefix, env_id)
            print(f"[{env_id}] uploading to {repo_id} ...")
            _push_folder_to_hub(
                folder_path=env_dir,
                repo_id=repo_id,
                private=args.private,
                token=args.hf_token,
                commit_message=f"Export normalized parquet for {env_id}",
                upload_method=args.upload_method,
                upload_num_workers=args.upload_num_workers,
                upload_retries=args.upload_retries,
                retry_backoff_sec=args.retry_backoff_sec,
            )
            print(f"[{env_id}] upload complete")
        elif args.push_to_hub and args.push_layout == "single-repo":
            repo_id = single_repo_id

        summary_entry = {
            "env_id": env_id,
            "output_dir": str(env_dir),
            "repo_id": repo_id,
            "tables": table_names,
            "stats": asdict(stats),
        }
        summary.append(summary_entry)

    if args.push_to_hub and args.push_layout == "single-repo":
        assert single_repo_id is not None
        _write_root_dataset_card(output_path, single_repo_id, summary)
        print(f"[all-envs] uploading to {single_repo_id} ...")
        _push_folder_to_hub(
            folder_path=output_path,
            repo_id=single_repo_id,
            private=args.private,
            token=args.hf_token,
            commit_message=f"Export normalized parquet for {len(summary)} environments",
            upload_method=args.upload_method,
            upload_num_workers=args.upload_num_workers,
            upload_retries=args.upload_retries,
            retry_backoff_sec=args.retry_backoff_sec,
        )
        print("[all-envs] upload complete")

    summary_path = output_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "push_layout": args.push_layout if args.push_to_hub else None,
                "single_repo_id": single_repo_id,
                "environments": summary,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
