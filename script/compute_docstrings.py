#!/usr/bin/env python3
"""
LLM annotation pipeline for the chunk_ter JSONL format.

Input (per env file, one JSON object per line):
- source metadata and TOC
- chunks: [{chunk_id, chunk_doc, chunk_elements}, ...]
- cache_field: {uid: docstring} already prefilled by chunk_ter reuse

Output (per env file, one JSON object per line):
- original entry
- one_liner
- header
- annotations (cache_field + newly generated docstrings)

The script is resumable:
- per-entry state is stored in --state-dir
- per-entry lock files prevent duplicate work across concurrent workers/processes
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml
from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(level: str, log_file: Optional[Path]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def clean_json_response(response_text: str) -> str:
    pattern = r"^```(?:json)?\s*(.*)\s*```$"
    match = re.match(pattern, response_text.strip(), re.DOTALL)
    return match.group(1) if match else response_text


def clean_coq_response(response_text: str) -> str:
    pattern = r"^```(?:coq)?\s*(.*)\s*```$"
    match = re.match(pattern, response_text.strip(), re.DOTALL)
    return match.group(1) if match else response_text


def source_relative_without_ext(source_path: str, library: str) -> str:
    normalized = source_path.replace("\\", "/")
    markers = [f"/{library}/"]
    if library == "Corelib":
        markers = [
            "/theories/",
            "/user-contrib/Corelib_duplicate/",
            "/Corelib/",
        ]

    rel = None
    for marker in markers:
        if marker in normalized:
            rel = normalized.split(marker, 1)[1]
            break
    if rel is None:
        rel = PurePosixPath(normalized).name

    rel_path = PurePosixPath(rel)
    if rel_path.suffix:
        rel_path = rel_path.with_suffix("")
    return rel_path.as_posix()


def entry_key(entry: Mapping[str, Any]) -> str:
    library = str(entry.get("library", ""))
    source = entry.get("source")
    if not isinstance(source, Mapping):
        source_path = ""
        source_content = ""
    else:
        source_path = str(source.get("path", ""))
        source_content = str(source.get("content", ""))

    digest = hashlib.sha256(source_content.encode("utf-8")).hexdigest()
    raw = f"{library}|{source_path}|{digest}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def source_content_hash(entry: Mapping[str, Any]) -> str:
    source = entry.get("source")
    if not isinstance(source, Mapping):
        return ""
    content = source.get("content")
    if not isinstance(content, str):
        return ""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def file_identity(entry: Mapping[str, Any]) -> Optional[Tuple[str, str, str]]:
    library = entry.get("library")
    source = entry.get("source")
    if not isinstance(library, str) or not isinstance(source, Mapping):
        return None
    source_path = source.get("path")
    if not isinstance(source_path, str):
        return None
    rel_no_ext = source_relative_without_ext(source_path, library)
    full_rel = f"{library}/{rel_no_ext}"
    node_id = f"file:{full_rel}"
    return library, full_rel, node_id


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_non_empty_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _as_string_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key, text in value.items():
        if not isinstance(key, str):
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        out[key] = text
    return out


@dataclass
class EntryRunStats:
    llm_calls: int = 0
    annotated_uids: int = 0
    cache_uids: int = 0
    generated_header: bool = False
    generated_one_liner: bool = False
    skipped_locked: bool = False
    had_error: bool = False


@dataclass
class EnvRunStats:
    env: str
    entries: int = 0
    llm_calls: int = 0
    annotated_uids: int = 0
    cache_uids: int = 0
    generated_headers: int = 0
    generated_one_liners: int = 0
    skipped_locked: int = 0
    errors: int = 0
    toc_dir_count: int = 0
    toc_file_count: int = 0
    toc_llm_calls: int = 0
    toc_reused_dirs: int = 0

    def add(self, entry_stats: EntryRunStats) -> None:
        self.entries += 1
        self.llm_calls += entry_stats.llm_calls
        self.annotated_uids += entry_stats.annotated_uids
        self.cache_uids += entry_stats.cache_uids
        self.generated_headers += int(entry_stats.generated_header)
        self.generated_one_liners += int(entry_stats.generated_one_liner)
        self.skipped_locked += int(entry_stats.skipped_locked)
        self.errors += int(entry_stats.had_error)


@dataclass
class TOCRunStats:
    directories: int = 0
    files: int = 0
    llm_calls: int = 0
    reused_directories: int = 0


@dataclass
class FileTOCRecord:
    entry_key: str
    node_id: str
    rel_path: str
    name: str
    one_liner: str
    annotation_count: int


class LLMAnnotator:
    def __init__(self, config_path: Path):
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.params: Dict[str, Any] = config.get("parameters", {}) or {}
        self.max_retries = int(config.get("max_retries", 0))

    def call(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            messages=messages,
            **self.params,
        )
        return response.choices[0].message.content or ""


class StateStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _env_dir(self, env_name: str) -> Path:
        path = self.root / env_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def state_path(self, env_name: str, key: str) -> Path:
        return self._env_dir(env_name) / f"{key}.json"

    def lock_path(self, env_name: str, key: str) -> Path:
        return self._env_dir(env_name) / f"{key}.lock"

    def load(self, env_name: str, key: str) -> Dict[str, Any]:
        path = self.state_path(env_name, key)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Ignoring corrupted state file: %s", path)
            return {}

    def save(self, env_name: str, key: str, state: Mapping[str, Any]) -> None:
        path = self.state_path(env_name, key)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)


@contextlib.contextmanager
def entry_lock(lock_path: Path, stale_seconds: int) -> Iterator[bool]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age > stale_seconds:
            logger.warning("Removing stale lock: %s", lock_path)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
        else:
            yield False
            return

    fd: Optional[int] = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} started={now_iso()}\n".encode("utf-8"))
        yield True
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class EntryProcessor:
    def __init__(
        self,
        annotator: LLMAnnotator,
        templates_dir: Path,
        state_store: StateStore,
        lock_stale_seconds: int,
    ):
        self.annotator = annotator
        self.state_store = state_store
        self.lock_stale_seconds = lock_stale_seconds

        self.header_template = (templates_dir / "prompt_header.txt").read_text(encoding="utf-8")
        self.one_liner_template = (templates_dir / "prompt_one_liner.txt").read_text(encoding="utf-8")
        self.task_template = (templates_dir / "prompt_docstrings.txt").read_text(encoding="utf-8")

    def _is_context_too_long_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "context length" in msg
            or "input (" in msg and "tokens" in msg and "longer than the model's context length" in msg
            or "maximum context length" in msg
        )

    def _generate_with_halving_source(
        self,
        *,
        template: str,
        source_content: str,
        field_name: str,
        env_name: str,
        line_idx: int,
        stats: EntryRunStats,
    ) -> str:
        current = source_content
        while True:
            prompt = template.replace("{source}", current)
            try:
                stats.llm_calls += 1
                response = self.annotator.call([{"role": "user", "content": prompt}])
                if len(current) < len(source_content):
                    logger.info(
                        "%s line %d: generated %s after trimming source (%d -> %d chars)",
                        env_name,
                        line_idx,
                        field_name,
                        len(source_content),
                        len(current),
                    )
                return clean_coq_response(response)
            except Exception as exc:
                if not self._is_context_too_long_error(exc):
                    raise
                if len(current) <= 1:
                    raise
                new_len = max(len(current) // 2, 1)
                if new_len == len(current):
                    raise
                logger.warning(
                    "%s line %d: %s prompt too long, retrying with half source (%d -> %d chars)",
                    env_name,
                    line_idx,
                    field_name,
                    len(current),
                    new_len,
                )
                current = current[:new_len]

    def _generate_one_liner(
        self,
        *,
        source_content: str,
        env_name: str,
        line_idx: int,
        stats: EntryRunStats,
    ) -> str:
        return self._generate_with_halving_source(
            template=self.one_liner_template,
            source_content=source_content,
            field_name="one_liner",
            env_name=env_name,
            line_idx=line_idx,
            stats=stats,
        )

    def _generate_header(
        self,
        *,
        source_content: str,
        env_name: str,
        line_idx: int,
        stats: EntryRunStats,
    ) -> str:
        return self._generate_with_halving_source(
            template=self.header_template,
            source_content=source_content,
            field_name="header",
            env_name=env_name,
            line_idx=line_idx,
            stats=stats,
        )

    def _parse_docstring_mapping(self, text: str, allowed: set[str]) -> Dict[str, str]:
        cleaned = clean_json_response(text)
        payload = json.loads(cleaned)
        if isinstance(payload, Mapping):
            if "items" in payload and isinstance(payload["items"], Sequence):
                out: Dict[str, str] = {}
                for item in payload["items"]:
                    if not isinstance(item, Mapping):
                        continue
                    uid = item.get("uid")
                    doc = item.get("docstring")
                    if isinstance(uid, str) and isinstance(doc, str) and uid in allowed:
                        out[uid] = doc
                return out
            out = {}
            for uid, doc in payload.items():
                if isinstance(uid, str) and isinstance(doc, str) and uid in allowed:
                    out[uid] = doc
            return out
        return {}

    def _annotate_chunk(
        self,
        header_context: str,
        chunk_doc: str,
        missing_uids: List[str],
        stats: EntryRunStats,
    ) -> Dict[str, str]:
        pending = list(missing_uids)
        final: Dict[str, str] = {}
        attempts = 0
        max_attempts = self.annotator.max_retries + 1

        while pending and attempts < max_attempts:
            attempts += 1
            missing_list_str = "\n".join(f"- {uid}" for uid in pending)
            prompt = (
                self.task_template
                .replace("{header}", header_context)
                .replace("{source}", chunk_doc)
                .replace("{missing}", missing_list_str)
            )
            try:
                stats.llm_calls += 1
                raw = self.annotator.call([{"role": "user", "content": prompt}])
                parsed = self._parse_docstring_mapping(raw, allowed=set(pending))
            except Exception as exc:
                logger.warning("Chunk annotation call failed (attempt %d/%d): %s", attempts, max_attempts, exc)
                continue

            for uid in list(pending):
                if uid in parsed:
                    final[uid] = parsed[uid]
                    pending.remove(uid)

            if pending:
                logger.info("Chunk still missing %d uid(s); retrying.", len(pending))

        if pending:
            logger.warning("Failed to annotate %d uid(s): %s", len(pending), sorted(pending))
        return final

    def process_entry(
        self,
        env_name: str,
        line_idx: int,
        entry: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], EntryRunStats]:
        stats = EntryRunStats()
        key = entry_key(entry)
        lock_path = self.state_store.lock_path(env_name, key)

        with entry_lock(lock_path, stale_seconds=self.lock_stale_seconds) as acquired:
            if not acquired:
                stats.skipped_locked = True
                state = self.state_store.load(env_name, key)
                return self._merge_output_entry(entry, state), stats

            state = self._build_or_load_state(env_name, key, entry)
            changed = False

            annotations = state.setdefault("annotations", {})
            if not isinstance(annotations, Mapping):
                annotations = {}
            annotations = dict(annotations)

            cache_field = entry.get("cache_field")
            if isinstance(cache_field, Mapping):
                for uid, doc in cache_field.items():
                    if not isinstance(uid, str) or not isinstance(doc, str):
                        continue
                    if uid not in annotations:
                        annotations[uid] = doc
                        stats.cache_uids += 1
                        changed = True
            state["annotations"] = annotations

            source = entry.get("source")
            if not isinstance(source, Mapping):
                raise ValueError("missing or invalid source")
            source_content = source.get("content")
            source_path = source.get("path")
            library = entry.get("library")
            if not isinstance(source_content, str) or not isinstance(source_path, str) or not isinstance(library, str):
                raise ValueError("missing source fields or library")

            if not state.get("one_liner"):
                try:
                    state["one_liner"] = self._generate_one_liner(
                        source_content=source_content,
                        env_name=env_name,
                        line_idx=line_idx,
                        stats=stats,
                    )
                    stats.generated_one_liner = True
                    changed = True
                except Exception as exc:
                    logger.error("%s line %d: failed to generate one_liner: %s", env_name, line_idx, exc)
                    stats.had_error = True

            if not state.get("header"):
                try:
                    state["header"] = self._generate_header(
                        source_content=source_content,
                        env_name=env_name,
                        line_idx=line_idx,
                        stats=stats,
                    )
                    stats.generated_header = True
                    changed = True
                except Exception as exc:
                    logger.error("%s line %d: failed to generate header: %s", env_name, line_idx, exc)
                    stats.had_error = True

            header = state.get("header")
            if isinstance(header, str) and header:
                chunks = entry.get("chunks")
                if isinstance(chunks, Sequence):
                    for chunk in chunks:
                        if not isinstance(chunk, Mapping):
                            continue
                        chunk_doc = chunk.get("chunk_doc")
                        if not isinstance(chunk_doc, str):
                            continue

                        chunk_uids: List[str] = []
                        seen: set[str] = set()
                        elements = chunk.get("chunk_elements")
                        if isinstance(elements, Sequence):
                            for element in elements:
                                if not isinstance(element, Mapping):
                                    continue
                                data = element.get("data")
                                if not isinstance(data, Mapping):
                                    continue
                                uid = data.get("uid")
                                if isinstance(uid, str) and uid not in seen:
                                    seen.add(uid)
                                    chunk_uids.append(uid)

                        missing = [uid for uid in chunk_uids if uid not in annotations]
                        if not missing:
                            continue

                        new_map = self._annotate_chunk(
                            header_context=header,
                            chunk_doc=chunk_doc,
                            missing_uids=missing,
                            stats=stats,
                        )
                        for uid, doc in new_map.items():
                            if uid not in annotations:
                                annotations[uid] = doc
                                stats.annotated_uids += 1
                                changed = True

                state["annotations"] = annotations
            else:
                logger.warning("%s line %d: missing header, skipping chunk annotation", env_name, line_idx)

            state["updated_at"] = now_iso()
            if changed:
                self.state_store.save(env_name, key, state)

            return self._merge_output_entry(entry, state), stats

    def _build_or_load_state(self, env_name: str, key: str, entry: Mapping[str, Any]) -> Dict[str, Any]:
        state = self.state_store.load(env_name, key)
        entry_one_liner = _to_non_empty_string(entry.get("one_liner"))
        entry_header = _to_non_empty_string(entry.get("header"))
        entry_annotations = _as_string_map(entry.get("annotations"))

        if state:
            changed = False

            if not _to_non_empty_string(state.get("one_liner")) and entry_one_liner is not None:
                state["one_liner"] = entry_one_liner
                changed = True
            if not _to_non_empty_string(state.get("header")) and entry_header is not None:
                state["header"] = entry_header
                changed = True

            annotations = _as_string_map(state.get("annotations"))
            for uid, docstring in entry_annotations.items():
                if uid not in annotations:
                    annotations[uid] = docstring
                    changed = True
            state["annotations"] = annotations

            if changed:
                state["updated_at"] = now_iso()
                self.state_store.save(env_name, key, state)
            return state

        source = entry.get("source")
        if not isinstance(source, Mapping):
            source = {}
        return {
            "entry_key": key,
            "library": entry.get("library"),
            "source_path": source.get("path"),
            "source_sha256": source_content_hash(entry),
            "one_liner": entry_one_liner,
            "header": entry_header,
            "annotations": entry_annotations,
            "updated_at": now_iso(),
        }

    def _merge_output_entry(self, entry: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(entry)
        one_liner = state.get("one_liner")
        header = state.get("header")
        annotations = state.get("annotations")
        entry_key_value = state.get("entry_key")
        if isinstance(one_liner, str):
            out["one_liner"] = one_liner
        if isinstance(header, str):
            out["header"] = header
        if isinstance(annotations, Mapping):
            out["annotations"] = dict(annotations)
        if isinstance(entry_key_value, str):
            out["entry_key"] = entry_key_value
        ident = file_identity(entry)
        if ident is not None:
            _, _, node_id = ident
            out["toc_node_id"] = node_id
        return out


class TOCBuilder:
    def __init__(
        self,
        annotator: LLMAnnotator,
        dir_prompt_template: str,
        state_store: StateStore,
    ):
        self.annotator = annotator
        self.dir_prompt_template = dir_prompt_template
        self.state_store = state_store

    def _toc_state_path(self, env_name: str) -> Path:
        path = self.state_store.root / "_toc" / f"{env_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_toc_state(self, env_name: str) -> Dict[str, Any]:
        path = self._toc_state_path(env_name)
        if not path.exists():
            return {"directories": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                directories = payload.get("directories")
                if isinstance(directories, Mapping):
                    return {"directories": dict(directories)}
        except json.JSONDecodeError:
            logger.warning("Ignoring corrupted TOC state: %s", path)
        return {"directories": {}}

    def _save_toc_state(self, env_name: str, state: Mapping[str, Any]) -> None:
        path = self._toc_state_path(env_name)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)

    def _dir_node_id(self, rel_path: str) -> str:
        return f"dir:{rel_path}" if rel_path else "dir:ROOT"

    def _dir_signature(self, child_descriptors: Sequence[str]) -> str:
        blob = "\n".join(child_descriptors)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _generate_dir_one_liner(self, dir_path: str, child_descriptors: Sequence[str]) -> str:
        children_block = "\n".join(child_descriptors)
        prompt = (
            self.dir_prompt_template
            .replace("{dir_path}", dir_path if dir_path else "ROOT")
            .replace("{children}", children_block)
        )
        text = self.annotator.call([{"role": "user", "content": prompt}])
        cleaned = clean_coq_response(text).strip()
        cleaned = cleaned.replace("\n", " ").strip()
        return cleaned if cleaned else "Directory summary unavailable."

    def build(
        self,
        env_name: str,
        records: Sequence[FileTOCRecord],
        output_path: Path,
    ) -> TOCRunStats:
        stats = TOCRunStats()
        nodes: Dict[str, Dict[str, Any]] = {}
        nodes["dir:ROOT"] = {
            "id": "dir:ROOT",
            "type": "directory",
            "name": "ROOT",
            "path": "",
            "parent_id": None,
            "children_ids": [],
            "one_liner": "",
        }
        file_index: Dict[str, str] = {}

        def ensure_dir(path: str) -> str:
            node_id = self._dir_node_id(path)
            if node_id in nodes:
                return node_id
            name = "ROOT" if path == "" else PurePosixPath(path).name
            parent_path = str(PurePosixPath(path).parent) if path else ""
            if parent_path == ".":
                parent_path = ""
            parent_id = self._dir_node_id(parent_path)
            ensure_dir(parent_path)
            nodes[node_id] = {
                "id": node_id,
                "type": "directory",
                "name": name,
                "path": path,
                "parent_id": parent_id,
                "children_ids": [],
                "one_liner": "",
            }
            if node_id not in nodes[parent_id]["children_ids"]:
                nodes[parent_id]["children_ids"].append(node_id)
            return node_id

        for rec in records:
            rel_parts = PurePosixPath(rec.rel_path).parts
            if len(rel_parts) < 2:
                continue
            parent_path = str(PurePosixPath(*rel_parts[:-1]))
            parent_id = ensure_dir(parent_path)
            file_node = {
                "id": rec.node_id,
                "type": "file",
                "name": rec.name,
                "path": rec.rel_path,
                "parent_id": parent_id,
                "children_ids": [],
                "one_liner": rec.one_liner,
                "annotation_count": rec.annotation_count,
                "entry_key": rec.entry_key,
            }
            nodes[rec.node_id] = file_node
            nodes[parent_id]["children_ids"].append(rec.node_id)
            file_index[rec.entry_key] = rec.node_id

        toc_state = self._load_toc_state(env_name)
        dir_state = toc_state.get("directories")
        if not isinstance(dir_state, Mapping):
            dir_state = {}
        dir_state = dict(dir_state)

        directory_nodes = [
            node for node in nodes.values() if node.get("type") == "directory"
        ]
        directory_nodes.sort(
            key=lambda n: len(str(n.get("path", "")).split("/")) if n.get("path") else 0,
            reverse=True,
        )

        for dir_node in directory_nodes:
            dir_id = dir_node["id"]
            children = sorted(dir_node.get("children_ids", []))
            child_descriptors: List[str] = []
            for child_id in children:
                child = nodes.get(child_id)
                if not child:
                    continue
                child_type = child.get("type", "unknown")
                child_name = child.get("name", "")
                child_summary = child.get("one_liner", "")
                child_descriptors.append(f"- [{child_type}] {child_name}: {child_summary}")

            signature = self._dir_signature(child_descriptors)
            cached = dir_state.get(dir_id)
            cached_summary = None
            if isinstance(cached, Mapping) and cached.get("signature") == signature:
                summary = cached.get("one_liner")
                if isinstance(summary, str) and summary.strip():
                    cached_summary = summary.strip()

            if cached_summary is not None:
                dir_node["one_liner"] = cached_summary
                stats.reused_directories += 1
            else:
                stats.llm_calls += 1
                try:
                    summary = self._generate_dir_one_liner(
                        dir_path=str(dir_node.get("path", "")),
                        child_descriptors=child_descriptors,
                    )
                    dir_node["one_liner"] = summary
                except Exception as exc:
                    logger.warning("Failed to generate directory summary for %s: %s", dir_id, exc)
                    dir_node["one_liner"] = "Directory summary unavailable."

            dir_state[dir_id] = {
                "signature": signature,
                "one_liner": dir_node["one_liner"],
                "updated_at": now_iso(),
            }

        toc_state["directories"] = dir_state
        self._save_toc_state(env_name, toc_state)

        nodes_list = sorted(nodes.values(), key=lambda n: (n.get("type"), n.get("path", ""), n.get("name", "")))
        payload = {
            "env": env_name,
            "generated_at": now_iso(),
            "root_id": "dir:ROOT",
            "nodes": nodes_list,
            "file_index": file_index,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = output_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(output_path)

        stats.directories = sum(1 for n in nodes_list if n.get("type") == "directory")
        stats.files = sum(1 for n in nodes_list if n.get("type") == "file")
        return stats


def iter_env_files(input_dir: Path, selected_envs: Optional[set[str]] = None) -> List[Path]:
    files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl"}
    )
    if selected_envs is None:
        return files
    return [p for p in files if p.stem in selected_envs]


def process_env_file(
    env_file: Path,
    output_dir: Path,
    toc_output_dir: Path,
    config_path: Path,
    templates_dir: Path,
    dir_prompt_template: str,
    state_store: StateStore,
    lock_stale_seconds: int,
) -> EnvRunStats:
    env_name = env_file.stem
    stats = EnvRunStats(env=env_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{env_name}.jsonl"
    temp_output_path = output_path.with_suffix(".jsonl.tmp")

    annotator = LLMAnnotator(config_path)
    processor = EntryProcessor(
        annotator=annotator,
        templates_dir=templates_dir,
        state_store=state_store,
        lock_stale_seconds=lock_stale_seconds,
    )
    toc_builder = TOCBuilder(
        annotator=annotator,
        dir_prompt_template=dir_prompt_template,
        state_store=state_store,
    )
    toc_records: List[FileTOCRecord] = []

    with env_file.open("r", encoding="utf-8") as in_f, temp_output_path.open("w", encoding="utf-8") as out_f:
        for line_idx, line in enumerate(in_f, start=1):
            if not line.strip():
                continue
            entry: Optional[Mapping[str, Any]] = None
            try:
                entry = json.loads(line)
                if not isinstance(entry, Mapping):
                    raise ValueError("line is not a JSON object")
                out_entry, entry_stats = processor.process_entry(env_name, line_idx, entry)
            except Exception as exc:
                logger.error("%s line %d: %s", env_name, line_idx, exc)
                entry_stats = EntryRunStats(had_error=True)
                if "entry" in locals() and isinstance(entry, Mapping):
                    out_entry = dict(entry)
                else:
                    out_entry = {"_error": str(exc)}

            out_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
            stats.add(entry_stats)

            if isinstance(out_entry, Mapping):
                ident = file_identity(out_entry)
                out_entry_key = out_entry.get("entry_key")
                one_liner = out_entry.get("one_liner")
                annotations = out_entry.get("annotations")
                if (
                    ident is not None
                    and isinstance(out_entry_key, str)
                ):
                    _, full_rel, node_id = ident
                    one_liner_text = one_liner if isinstance(one_liner, str) and one_liner.strip() else "Missing one-liner"
                    annotation_count = len(annotations) if isinstance(annotations, Mapping) else 0
                    toc_records.append(
                        FileTOCRecord(
                            entry_key=out_entry_key,
                            node_id=node_id,
                            rel_path=full_rel,
                            name=PurePosixPath(full_rel).name,
                            one_liner=one_liner_text,
                            annotation_count=annotation_count,
                        )
                    )

    temp_output_path.replace(output_path)
    toc_path = toc_output_dir / f"{env_name}.toc.json"
    toc_stats = toc_builder.build(env_name=env_name, records=toc_records, output_path=toc_path)
    stats.toc_dir_count = toc_stats.directories
    stats.toc_file_count = toc_stats.files
    stats.toc_llm_calls = toc_stats.llm_calls
    stats.toc_reused_dirs = toc_stats.reused_directories

    logger.info(
        (
            "Processed %s -> %s | entries=%d llm_calls=%d annotated=%d "
            "cache=%d generated_header=%d "
            "generated_one_liner=%d locked=%d errors=%d toc_dirs=%d toc_files=%d toc_llm_calls=%d toc_reused=%d"
        ),
        env_file.name,
        output_path.name,
        stats.entries,
        stats.llm_calls,
        stats.annotated_uids,
        stats.cache_uids,
        stats.generated_headers,
        stats.generated_one_liners,
        stats.skipped_locked,
        stats.errors,
        stats.toc_dir_count,
        stats.toc_file_count,
        stats.toc_llm_calls,
        stats.toc_reused_dirs,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute docstrings from chunk_ter JSONL exports.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("chunk_ter_output"),
        help="Directory containing chunk_ter per-env JSON/JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotated_jsonl"),
        help="Directory where annotated per-env JSONL files are written.",
    )
    parser.add_argument(
        "--toc-output-dir",
        type=Path,
        default=None,
        help="Directory where per-env TOC JSON files are written (default: <output-dir>/toc).",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".compute_docstrings_state"),
        help="Directory for resumable per-entry state and lock files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/annotator/config.yaml"),
        help="LLM configuration YAML.",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("config/annotator"),
        help="Prompt templates directory.",
    )
    parser.add_argument(
        "--dir-template",
        type=Path,
        default=Path("config/annotator/prompt_dir_one_liner.txt"),
        help="Prompt template for directory one-liner generation.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment name (file stem). Repeat for multiple envs.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of env files processed in parallel.")
    parser.add_argument(
        "--lock-stale-seconds",
        type=int,
        default=4 * 60 * 60,
        help="Treat lock files older than this threshold as stale.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path.")
    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        logger.error("Invalid input directory: %s", args.input_dir)
        sys.exit(1)
    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    if not args.templates.exists():
        logger.error("Templates directory not found: %s", args.templates)
        sys.exit(1)
    if not args.dir_template.exists():
        logger.error("Directory prompt template not found: %s", args.dir_template)
        sys.exit(1)

    selected_envs = set(args.env) if args.env else None
    env_files = iter_env_files(args.input_dir, selected_envs=selected_envs)
    if not env_files:
        logger.warning("No JSON/JSONL files found in %s", args.input_dir)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    toc_output_dir = args.toc_output_dir or (args.output_dir / "toc")
    toc_output_dir.mkdir(parents=True, exist_ok=True)
    dir_prompt_template = args.dir_template.read_text(encoding="utf-8")
    state_store = StateStore(args.state_dir)

    logger.info("Found %d env files", len(env_files))
    logger.info("Input dir: %s", args.input_dir)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("TOC output dir: %s", toc_output_dir)
    logger.info("State dir: %s", args.state_dir)
    logger.info("Workers: %d", args.workers)

    all_stats: List[EnvRunStats] = []
    if args.workers <= 1:
        for env_file in tqdm(env_files, desc="Annotating env files"):
            all_stats.append(
                process_env_file(
                    env_file=env_file,
                    output_dir=args.output_dir,
                    toc_output_dir=toc_output_dir,
                    config_path=args.config,
                    templates_dir=args.templates,
                    dir_prompt_template=dir_prompt_template,
                    state_store=state_store,
                    lock_stale_seconds=args.lock_stale_seconds,
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_env_file,
                    env_file=env_file,
                    output_dir=args.output_dir,
                    toc_output_dir=toc_output_dir,
                    config_path=args.config,
                    templates_dir=args.templates,
                    dir_prompt_template=dir_prompt_template,
                    state_store=state_store,
                    lock_stale_seconds=args.lock_stale_seconds,
                )
                for env_file in env_files
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Annotating env files"):
                all_stats.append(future.result())

    total_entries = sum(s.entries for s in all_stats)
    total_llm_calls = sum(s.llm_calls for s in all_stats)
    total_annotated = sum(s.annotated_uids for s in all_stats)
    total_cache = sum(s.cache_uids for s in all_stats)
    total_errors = sum(s.errors for s in all_stats)
    total_locked = sum(s.skipped_locked for s in all_stats)
    total_toc_dirs = sum(s.toc_dir_count for s in all_stats)
    total_toc_files = sum(s.toc_file_count for s in all_stats)
    total_toc_llm_calls = sum(s.toc_llm_calls for s in all_stats)
    total_toc_reused = sum(s.toc_reused_dirs for s in all_stats)

    logger.info(
        (
            "Done. env_files=%d entries=%d llm_calls=%d annotated_uids=%d "
            "cache_uids=%d locked=%d errors=%d toc_dirs=%d toc_files=%d toc_llm_calls=%d toc_reused=%d"
        ),
        len(all_stats),
        total_entries,
        total_llm_calls,
        total_annotated,
        total_cache,
        total_locked,
        total_errors,
        total_toc_dirs,
        total_toc_files,
        total_toc_llm_calls,
        total_toc_reused,
    )


if __name__ == "__main__":
    main()
