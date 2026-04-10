#!/usr/bin/env python3
"""Pre-compute docstring embeddings directly from a Hugging Face dataset.

The script downloads one environment slice (or all environments) from HF
(streaming by default), extracts TOC docstrings, computes embeddings in
batches, and writes one `.pt` file per library.

Each exported text is: `<full source statement> + <docstring>`.
Localizations include both byte offsets (`start_bp`, `end_bp`) and derived
line/character positions computed from UTF-8 source content.

Supported HF row formats:
1. Raw rows containing `library`, `source`, `toc`.
2. `source_blobs` rows containing `entry_json` (serialized raw row).
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, TextIO, Tuple


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _non_empty_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _as_string_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key, doc in value.items():
        if not isinstance(key, str):
            continue
        text = _non_empty_str(doc)
        if text is not None:
            out[key] = text
    return out


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


def _parse_range(range_obj: Any) -> Dict[str, Optional[int]]:
    if not isinstance(range_obj, Mapping):
        return {
            "start_line": None,
            "start_character": None,
            "end_line": None,
            "end_character": None,
        }
    start = range_obj.get("start")
    end = range_obj.get("end")
    if not isinstance(start, Mapping):
        start = {}
    if not isinstance(end, Mapping):
        end = {}
    return {
        "start_line": _to_int(start.get("line")),
        "start_character": _to_int(start.get("character")),
        "end_line": _to_int(end.get("line")),
        "end_character": _to_int(end.get("character")),
    }


def _parse_span(span_obj: Any) -> Dict[str, Optional[int]]:
    if not isinstance(span_obj, Mapping):
        return {"start_bp": None, "end_bp": None}
    start_bp = _to_int(span_obj.get("start_bp"))
    end_bp = _to_int(span_obj.get("end_bp"))
    if start_bp is None:
        start_bp = _to_int(span_obj.get("bp"))
    if end_bp is None:
        end_bp = _to_int(span_obj.get("ep"))
    return {
        "start_bp": start_bp,
        "end_bp": end_bp,
    }


def _normalize_bp_ep(bp: Optional[int], ep: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(bp, int) and bp < 0:
        bp = 0
    if isinstance(ep, int) and ep < 0:
        ep = None
    if isinstance(bp, int) and isinstance(ep, int) and ep < bp:
        ep = bp
    return bp, ep


def _extract_bp_ep(row: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    start_bp = _to_int(row.get("start_bp"))
    end_bp = _to_int(row.get("end_bp"))
    if start_bp is None:
        start_bp = _to_int(row.get("span_bp"))
    if end_bp is None:
        end_bp = _to_int(row.get("span_ep"))

    if start_bp is None or end_bp is None:
        span_obj = row.get("span")
        if isinstance(span_obj, Mapping):
            parsed = _parse_span(span_obj)
            if start_bp is None:
                start_bp = parsed["start_bp"]
            if end_bp is None:
                end_bp = parsed["end_bp"]

    if start_bp is None:
        start_bp = _to_int(row.get("bp"))
    if end_bp is None:
        end_bp = _to_int(row.get("ep"))
    return _normalize_bp_ep(start_bp, end_bp)


def _build_line_start_offsets(content_bytes: bytes) -> List[int]:
    starts = [0]
    cursor = 0
    while True:
        idx = content_bytes.find(b"\n", cursor)
        if idx < 0:
            break
        starts.append(idx + 1)
        cursor = idx + 1
    return starts


@dataclass
class SourceTextView:
    content: str
    content_bytes: bytes
    line_starts: List[int]


def _build_source_text_view(source_content: Optional[str]) -> Optional[SourceTextView]:
    if not isinstance(source_content, str):
        return None
    content_bytes = source_content.encode("utf-8")
    return SourceTextView(
        content=source_content,
        content_bytes=content_bytes,
        line_starts=_build_line_start_offsets(content_bytes),
    )


def _byte_offset_to_line_char(content_bytes: bytes, line_starts: Sequence[int], offset: int) -> Tuple[int, int]:
    offset = max(0, min(offset, len(content_bytes)))
    line = bisect.bisect_right(line_starts, offset) - 1
    if line < 0:
        line = 0
    line_start = line_starts[line] if line < len(line_starts) else 0
    segment = content_bytes[line_start:offset]
    character = len(segment.decode("utf-8", errors="ignore"))
    return line, character


def _localization_from_offsets(
    source_view: Optional[SourceTextView],
    bp: Optional[int],
    ep: Optional[int],
    range_hint: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Optional[int]]:
    bp, ep = _normalize_bp_ep(bp, ep)
    out: Dict[str, Optional[int]] = {
        "start_bp": bp,
        "end_bp": ep,
        "start_line": None,
        "start_character": None,
        "end_line": None,
        "end_character": None,
    }

    if source_view is not None and (bp is not None or ep is not None):
        content_bytes = source_view.content_bytes
        line_starts = source_view.line_starts
        if bp is not None:
            start_line, start_char = _byte_offset_to_line_char(content_bytes, line_starts, bp)
            out["start_line"] = start_line
            out["start_character"] = start_char
        if ep is not None:
            end_line, end_char = _byte_offset_to_line_char(content_bytes, line_starts, ep)
            out["end_line"] = end_line
            out["end_character"] = end_char

    if isinstance(range_hint, Mapping):
        parsed = _parse_range(range_hint)
        if out["start_line"] is None:
            out["start_line"] = parsed["start_line"]
        if out["start_character"] is None:
            out["start_character"] = parsed["start_character"]
        if out["end_line"] is None:
            out["end_line"] = parsed["end_line"]
        if out["end_character"] is None:
            out["end_character"] = parsed["end_character"]

    return out


def _statement_from_offsets(source_view: Optional[SourceTextView], bp: Optional[int], ep: Optional[int]) -> str:
    if source_view is None:
        return ""
    bp, ep = _normalize_bp_ep(bp, ep)
    if bp is None or ep is None:
        return ""
    content_bytes = source_view.content_bytes
    bp = max(0, min(bp, len(content_bytes)))
    ep = max(bp, min(ep, len(content_bytes)))
    if ep <= bp:
        return ""
    return content_bytes[bp:ep].decode("utf-8", errors="replace").strip()


def _merge_statement_and_docstring(statement: str, docstring: str) -> str:
    statement = statement.strip()
    docstring = docstring.strip()
    if statement and docstring:
        return f"{statement}\n\n{docstring}"
    if statement:
        return statement
    return docstring


def _safe_filename(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or "library"


def _iter_with_first(first: Any, iterator: Iterator[Any]) -> Iterator[Any]:
    yield first
    for item in iterator:
        yield item


@dataclass
class DocRecord:
    library: str
    docstring: str
    docstring_raw: str
    statement: str
    logical_path: Optional[str]
    relative_path: str
    source_path: str
    uid: Optional[str]
    kind: Optional[str]
    name: Optional[str]
    localization: Dict[str, Optional[int]]


def _extract_doc_records(entry: Mapping[str, Any]) -> List[DocRecord]:
    library = _to_str(entry.get("library"))
    source = entry.get("source")
    if not isinstance(library, str) or not isinstance(source, Mapping):
        return []

    source_path = _to_str(source.get("path")) or ""
    logical_path = _to_str(source.get("logical_path"))
    relative_path = _source_relative_path(source_path, library)
    source_view = _build_source_text_view(_to_str(source.get("content")))

    annotations = _as_string_map(entry.get("annotations"))
    cache_field = _as_string_map(entry.get("cache_field"))
    toc = entry.get("toc")
    if not isinstance(toc, list):
        return []

    out: List[DocRecord] = []

    def walk(nodes: Sequence[Any]) -> None:
        for raw in nodes:
            if not isinstance(raw, Mapping):
                continue

            data = raw.get("data")
            if not isinstance(data, Mapping):
                data = {}

            uid = _to_str(data.get("uid"))
            doc_raw = _non_empty_str(raw.get("docstring"))
            if doc_raw is None and isinstance(uid, str):
                doc_raw = annotations.get(uid) or cache_field.get(uid)

            if doc_raw is not None:
                bp, ep = _extract_bp_ep(raw)
                localization = _localization_from_offsets(
                    source_view=source_view,
                    bp=bp,
                    ep=ep,
                    range_hint=raw.get("range"),
                )
                statement = _statement_from_offsets(source_view=source_view, bp=bp, ep=ep)
                embedding_text = _merge_statement_and_docstring(statement=statement, docstring=doc_raw)
                out.append(
                    DocRecord(
                        library=library,
                        docstring=embedding_text,
                        docstring_raw=doc_raw,
                        statement=statement,
                        logical_path=logical_path,
                        relative_path=relative_path,
                        source_path=source_path,
                        uid=uid,
                        kind=_to_str(raw.get("kind")),
                        name=_to_str(raw.get("name")),
                        localization=localization,
                    )
                )

            members = raw.get("members")
            if isinstance(members, list):
                walk(members)

    walk(toc)
    return out


def _detect_row_mode_from_keys(keys: set[str]) -> Optional[str]:
    if {"library", "source", "toc"}.issubset(keys):
        return "raw"
    if "entry_json" in keys:
        return "source_blobs"
    return None


def _looks_like_sources_table_keys(keys: set[str]) -> bool:
    required = {"library", "source_path", "logical_path"}
    return required.issubset(keys)


def _detect_row_mode_from_row(row: Any) -> Optional[str]:
    if not isinstance(row, Mapping):
        return None
    keys = set(row.keys())
    return _detect_row_mode_from_keys(keys)


def _load_dataset_once(
    dataset_id: str,
    config_name: str,
    split: str,
    token: Optional[str],
    revision: Optional[str],
    streaming: bool,
) -> Any:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: pip install datasets") from exc

    kwargs: Dict[str, Any] = {"split": split, "streaming": streaming}
    if token is not None:
        kwargs["token"] = token
    if revision is not None:
        kwargs["revision"] = revision

    try:
        return load_dataset(dataset_id, config_name, **kwargs)
    except TypeError:
        if "token" in kwargs:
            kwargs["use_auth_token"] = kwargs.pop("token")
        return load_dataset(dataset_id, config_name, **kwargs)


def _select_hf_rows(
    dataset_id: str,
    env: str,
    config_override: Optional[str],
    split: str,
    token: Optional[str],
    revision: Optional[str],
    streaming: bool,
) -> Tuple[str, str, Any]:
    if config_override is not None:
        candidates = [config_override]
    else:
        candidates = [f"{env}-source_blobs", env, f"{env}-sources"]

    errors: List[str] = []
    for config_name in candidates:
        try:
            dataset = _load_dataset_once(
                dataset_id=dataset_id,
                config_name=config_name,
                split=split,
                token=token,
                revision=revision,
                streaming=streaming,
            )
        except Exception as exc:
            errors.append(f"{config_name}: {type(exc).__name__}: {exc}")
            continue

        mode: Optional[str] = None
        features = getattr(dataset, "features", None)
        if features is not None:
            try:
                mode = _detect_row_mode_from_keys(set(features.keys()))
            except Exception:
                mode = None

        if mode is not None:
            return config_name, mode, iter(dataset)

        features = getattr(dataset, "features", None)
        if features is not None:
            try:
                keys = set(features.keys())
                if _looks_like_sources_table_keys(keys):
                    toc_config = f"{env}-toc_nodes"
                    toc_dataset = _load_dataset_once(
                        dataset_id=dataset_id,
                        config_name=toc_config,
                        split=split,
                        token=token,
                        revision=revision,
                        streaming=streaming,
                    )
                    return (
                        f"{config_name}+{toc_config}",
                        "table_only",
                        {"sources": iter(dataset), "toc_nodes": iter(toc_dataset)},
                    )
            except Exception as exc:
                errors.append(f"{config_name}: table-only pair load failed: {type(exc).__name__}: {exc}")
                continue

        it = iter(dataset)
        try:
            first = next(it)
        except StopIteration:
            errors.append(f"{config_name}: dataset split is empty")
            continue

        mode = _detect_row_mode_from_row(first)
        if mode is not None:
            return config_name, mode, _iter_with_first(first, it)

        if isinstance(first, Mapping) and _looks_like_sources_table_keys(set(first.keys())):
            try:
                toc_config = f"{env}-toc_nodes"
                toc_dataset = _load_dataset_once(
                    dataset_id=dataset_id,
                    config_name=toc_config,
                    split=split,
                    token=token,
                    revision=revision,
                    streaming=streaming,
                )
                return (
                    f"{config_name}+{toc_config}",
                    "table_only",
                    {"sources": _iter_with_first(first, it), "toc_nodes": iter(toc_dataset)},
                )
            except Exception as exc:
                errors.append(f"{config_name}: table-only pair load failed: {type(exc).__name__}: {exc}")
                continue

        row_keys = sorted(first.keys()) if isinstance(first, Mapping) else [type(first).__name__]
        errors.append(f"{config_name}: unsupported row schema keys={row_keys}")
        continue

    msg = "\n".join(f"- {e}" for e in errors) if errors else "- no candidate config matched"
    raise RuntimeError(
        "Could not load a supported HF split for this env.\n"
        "Expected one of:\n"
        "- raw rows (library/source/toc)\n"
        "- source_blobs rows (entry_json)\n"
        "- table pair configs (<env>-sources + <env>-toc_nodes)\n"
        f"Tried:\n{msg}"
    )


def _decode_hf_row(row: Mapping[str, Any], mode: str, row_idx: int) -> Optional[Dict[str, Any]]:
    if mode == "raw":
        return dict(row)
    if mode == "source_blobs":
        raw_entry = row.get("entry_json")
        if not isinstance(raw_entry, str):
            raise RuntimeError(f"row #{row_idx} missing string entry_json")
        try:
            payload = json.loads(raw_entry)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"row #{row_idx} has invalid entry_json: {exc}") from exc
        if not isinstance(payload, dict):
            return None
        return payload
    raise RuntimeError(f"Unsupported row mode: {mode}")


TABLE_CONFIG_SUFFIXES = {
    "sources",
    "toc_nodes",
    "proofs",
    "proof_steps",
    "step_deps",
    "proof_axioms",
    "env_toc",
    "env_metadata",
    "source_blobs",
    "diags",
}


def _split_table_config(config_name: str) -> Optional[Tuple[str, str]]:
    for suffix in sorted(TABLE_CONFIG_SUFFIXES, key=len, reverse=True):
        marker = f"-{suffix}"
        if config_name.endswith(marker) and len(config_name) > len(marker):
            env = config_name[: -len(marker)]
            return env, suffix
    return None


def _load_dataset_config_names(
    dataset_id: str,
    token: Optional[str],
    revision: Optional[str],
) -> List[str]:
    try:
        from datasets import get_dataset_config_names  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: pip install datasets") from exc

    kwargs: Dict[str, Any] = {}
    if token is not None:
        kwargs["token"] = token
    if revision is not None:
        kwargs["revision"] = revision

    try:
        names = get_dataset_config_names(dataset_id, **kwargs)
    except TypeError:
        if "token" in kwargs:
            kwargs["use_auth_token"] = kwargs.pop("token")
        names = get_dataset_config_names(dataset_id, **kwargs)
    return [str(name) for name in names]


def _looks_like_table_config(config_name: str) -> bool:
    return _split_table_config(config_name) is not None


def _discover_envs(
    dataset_id: str,
    token: Optional[str],
    revision: Optional[str],
) -> List[str]:
    configs = _load_dataset_config_names(dataset_id=dataset_id, token=token, revision=revision)
    envs: set[str] = set()
    table_by_env: Dict[str, set[str]] = defaultdict(set)

    for config_name in configs:
        split = _split_table_config(config_name)
        if split is not None:
            env_name, suffix = split
            table_by_env[env_name].add(suffix)

    # Preferred: explicit lossless configs.
    for config_name in configs:
        split = _split_table_config(config_name)
        if split is not None and split[1] == "source_blobs":
            envs.add(split[0])

    # Also include non-table configs (typically raw env configs).
    for config_name in configs:
        if config_name in envs:
            continue
        if _looks_like_table_config(config_name):
            continue
        envs.add(config_name)

    # Table-only fallback: require at least sources + toc_nodes.
    for env_name, suffixes in table_by_env.items():
        if "sources" in suffixes and "toc_nodes" in suffixes:
            envs.add(env_name)

    out = sorted(envs)
    if not out:
        raise RuntimeError(
            "Could not discover environments from dataset configs. "
            "Expected configs like '<env>-source_blobs', raw '<env>', "
            "or table pairs '<env>-sources' + '<env>-toc_nodes'."
        )
    return out


def _default_device() -> str:
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(torch_module: Any, dtype: str) -> Optional[Any]:
    if dtype == "auto":
        return None
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    return mapping[dtype]


class HFEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int,
        trust_remote_code: bool,
        normalize: bool,
        torch_dtype: str,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependencies. Install with: pip install torch transformers"
            ) from exc

        self.torch = torch
        self.normalize = normalize
        self.max_length = max_length
        self.device = torch.device(device)

        dtype = _resolve_torch_dtype(torch, torch_dtype)
        model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _mean_pool(self, token_embeddings: Any, attention_mask: Optional[Any]) -> Any:
        if attention_mask is None:
            return token_embeddings.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        return summed / denom

    def _from_mapping(self, outputs: Mapping[str, Any], attention_mask: Optional[Any]) -> Any:
        for key in ("sentence_embedding", "embeddings", "pooler_output"):
            value = outputs.get(key)
            if self.torch.is_tensor(value):
                return value
        last_hidden = outputs.get("last_hidden_state")
        if self.torch.is_tensor(last_hidden):
            return self._mean_pool(last_hidden, attention_mask)
        raise RuntimeError("Could not find embedding tensor in model outputs.")

    def _from_object(self, outputs: Any, attention_mask: Optional[Any]) -> Any:
        for key in ("sentence_embedding", "embeddings", "pooler_output"):
            value = getattr(outputs, key, None)
            if self.torch.is_tensor(value):
                return value
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if self.torch.is_tensor(last_hidden):
            return self._mean_pool(last_hidden, attention_mask)
        if isinstance(outputs, (tuple, list)) and outputs:
            first = outputs[0]
            if self.torch.is_tensor(first):
                if first.dim() == 3:
                    return self._mean_pool(first, attention_mask)
                return first
        raise RuntimeError("Unsupported output type while extracting embeddings.")

    def encode(self, texts: Sequence[str]) -> Any:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with self.torch.inference_mode():
            outputs = self.model(**encoded)

        attention_mask = encoded.get("attention_mask")
        if self.torch.is_tensor(outputs):
            embeddings = outputs
        elif isinstance(outputs, Mapping):
            embeddings = self._from_mapping(outputs, attention_mask)
        else:
            embeddings = self._from_object(outputs, attention_mask)

        if embeddings.dim() == 3:
            embeddings = self._mean_pool(embeddings, attention_mask)
        if embeddings.dim() != 2:
            raise RuntimeError(f"Expected 2D embeddings, got shape={tuple(embeddings.shape)}")

        embeddings = embeddings.to(device="cpu", dtype=self.torch.float32)
        if self.normalize and embeddings.numel() > 0:
            embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


def _open_library_writer(
    library: str,
    temp_dir: Path,
    handles: Dict[str, TextIO],
    path_by_library: Dict[str, Path],
    used_paths: set[Path],
) -> TextIO:
    handle = handles.get(library)
    if handle is not None:
        return handle

    stem = _safe_filename(library)
    path = temp_dir / f"{stem}.jsonl"
    if path in used_paths:
        idx = 1
        while True:
            candidate = temp_dir / f"{stem}_{idx}.jsonl"
            if candidate not in used_paths:
                path = candidate
                break
            idx += 1

    used_paths.add(path)
    path_by_library[library] = path
    handle = path.open("w", encoding="utf-8")
    handles[library] = handle
    return handle


def _spool_docstrings_by_library(
    rows: Iterable[Mapping[str, Any]],
    row_mode: str,
    temp_dir: Path,
) -> Tuple[Dict[str, Path], Dict[str, int], int]:
    handles: Dict[str, TextIO] = {}
    path_by_library: Dict[str, Path] = {}
    used_paths: set[Path] = set()
    counts: Dict[str, int] = defaultdict(int)
    total = 0

    try:
        for row_idx, row in enumerate(rows, start=1):
            entry = _decode_hf_row(row=row, mode=row_mode, row_idx=row_idx)
            if entry is None:
                continue
            records = _extract_doc_records(entry)
            for record in records:
                handle = _open_library_writer(
                    library=record.library,
                    temp_dir=temp_dir,
                    handles=handles,
                    path_by_library=path_by_library,
                    used_paths=used_paths,
                )
                handle.write(
                    json.dumps(
                        {
                            "docstring": record.docstring,
                            "docstring_raw": record.docstring_raw,
                            "statement": record.statement,
                            "logical_path": record.logical_path,
                            "relative_path": record.relative_path,
                            "source_path": record.source_path,
                            "uid": record.uid,
                            "kind": record.kind,
                            "name": record.name,
                            "localization": record.localization,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                counts[record.library] += 1
                total += 1
            if row_idx % 1000 == 0:
                print(f"[scan] rows={row_idx} docstrings={total}")
    finally:
        for handle in handles.values():
            handle.close()

    return path_by_library, dict(counts), total


def _source_meta_from_table_row(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    library = _to_str(row.get("library"))
    source_path = _to_str(row.get("source_path"))
    if not isinstance(library, str) or not isinstance(source_path, str):
        return None
    source_content = _to_str(row.get("content"))

    return {
        "library": library,
        "source_path": source_path,
        "logical_path": _to_str(row.get("logical_path")),
        "relative_path": _source_relative_path(source_path, library),
        "source_id": _to_str(row.get("source_id")),
        "source_view": _build_source_text_view(source_content),
    }


def _toc_localization_from_table_row(
    row: Mapping[str, Any],
    source_view: Optional[SourceTextView],
) -> Dict[str, Optional[int]]:
    start_bp, end_bp = _extract_bp_ep(row)
    return _localization_from_offsets(
        source_view=source_view,
        bp=start_bp,
        ep=end_bp,
        range_hint={
            "start": {
                "line": _to_int(row.get("range_start_line")),
                "character": _to_int(row.get("range_start_character")),
            },
            "end": {
                "line": _to_int(row.get("range_end_line")),
                "character": _to_int(row.get("range_end_character")),
            },
        },
    )


def _write_spooled_record(
    handle: TextIO,
    library: str,
    source_path: str,
    logical_path: Optional[str],
    relative_path: str,
    docstring: str,
    docstring_raw: str,
    statement: str,
    uid: Optional[str],
    kind: Optional[str],
    name: Optional[str],
    localization: Dict[str, Optional[int]],
) -> None:
    handle.write(
        json.dumps(
            {
                "docstring": docstring,
                "docstring_raw": docstring_raw,
                "statement": statement,
                "logical_path": logical_path,
                "relative_path": relative_path,
                "source_path": source_path,
                "uid": uid,
                "kind": kind,
                "name": name,
                "localization": localization,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def _spool_docstrings_by_library_from_tables(
    sources_rows: Iterable[Mapping[str, Any]],
    toc_rows: Iterable[Mapping[str, Any]],
    temp_dir: Path,
) -> Tuple[Dict[str, Path], Dict[str, int], int, Dict[str, Any]]:
    handles: Dict[str, TextIO] = {}
    path_by_library: Dict[str, Path] = {}
    used_paths: set[Path] = set()
    counts: Dict[str, int] = defaultdict(int)
    total = 0

    source_list: List[Dict[str, Any]] = []
    source_by_id: Dict[str, Dict[str, Any]] = {}
    for src_row in sources_rows:
        if not isinstance(src_row, Mapping):
            continue
        meta = _source_meta_from_table_row(src_row)
        if meta is None:
            continue
        source_list.append(meta)
        source_id = meta.get("source_id")
        if isinstance(source_id, str):
            source_by_id[source_id] = meta

    if not source_list:
        return {}, {}, 0, {"table_link_mode": "none", "num_sources": 0, "num_toc_rows": 0}

    toc_it = iter(toc_rows)
    try:
        first_toc = next(toc_it)
    except StopIteration:
        return {}, {}, 0, {"table_link_mode": "none", "num_sources": len(source_list), "num_toc_rows": 0}
    if not isinstance(first_toc, Mapping):
        first_toc = {}
    all_toc_rows = _iter_with_first(first_toc, toc_it)

    has_toc_source_id = isinstance(first_toc, Mapping) and "source_id" in first_toc
    link_mode = "by_source_id" if has_toc_source_id and source_by_id else "by_preorder_segments"
    toc_count = 0
    source_advance_events = 0
    source_overflow = False

    try:
        if link_mode == "by_source_id":
            for toc_idx, toc_row in enumerate(all_toc_rows, start=1):
                if not isinstance(toc_row, Mapping):
                    continue
                toc_count += 1
                docstring = _non_empty_str(toc_row.get("docstring"))
                if docstring is None:
                    continue

                source_id = _to_str(toc_row.get("source_id"))
                if not isinstance(source_id, str):
                    continue
                source_meta = source_by_id.get(source_id)
                if source_meta is None:
                    continue

                library = source_meta["library"]
                source_path = source_meta["source_path"]
                relative_path = source_meta["relative_path"]
                if not isinstance(library, str) or not isinstance(source_path, str) or not isinstance(relative_path, str):
                    continue
                source_view = source_meta.get("source_view")
                localization = _toc_localization_from_table_row(toc_row, source_view)
                statement = _statement_from_offsets(
                    source_view=source_view,
                    bp=localization.get("start_bp"),
                    ep=localization.get("end_bp"),
                )
                embedding_text = _merge_statement_and_docstring(statement=statement, docstring=docstring)

                handle = _open_library_writer(
                    library=library,
                    temp_dir=temp_dir,
                    handles=handles,
                    path_by_library=path_by_library,
                    used_paths=used_paths,
                )
                _write_spooled_record(
                    handle=handle,
                    library=library,
                    source_path=source_path,
                    logical_path=source_meta.get("logical_path"),
                    relative_path=relative_path,
                    docstring=embedding_text,
                    docstring_raw=docstring,
                    statement=statement,
                    uid=_to_str(toc_row.get("data_uid")),
                    kind=_to_str(toc_row.get("kind")),
                    name=_to_str(toc_row.get("name")),
                    localization=localization,
                )
                counts[library] += 1
                total += 1
                if toc_idx % 10000 == 0:
                    print(f"[scan] toc_rows={toc_idx} docstrings={total}")
        else:
            source_pos = 0
            current_source = source_list[source_pos]
            has_started = False

            for toc_idx, toc_row in enumerate(all_toc_rows, start=1):
                if not isinstance(toc_row, Mapping):
                    continue
                toc_count += 1
                preorder = _to_int(toc_row.get("preorder_index"))
                if preorder == 0:
                    if has_started:
                        source_pos += 1
                        source_advance_events += 1
                        if source_pos >= len(source_list):
                            source_overflow = True
                            break
                        current_source = source_list[source_pos]
                    else:
                        has_started = True

                if not has_started:
                    has_started = True

                docstring = _non_empty_str(toc_row.get("docstring"))
                if docstring is None:
                    continue

                library = current_source["library"]
                source_path = current_source["source_path"]
                relative_path = current_source["relative_path"]
                if not isinstance(library, str) or not isinstance(source_path, str) or not isinstance(relative_path, str):
                    continue
                source_view = current_source.get("source_view")
                localization = _toc_localization_from_table_row(toc_row, source_view)
                statement = _statement_from_offsets(
                    source_view=source_view,
                    bp=localization.get("start_bp"),
                    ep=localization.get("end_bp"),
                )
                embedding_text = _merge_statement_and_docstring(statement=statement, docstring=docstring)

                handle = _open_library_writer(
                    library=library,
                    temp_dir=temp_dir,
                    handles=handles,
                    path_by_library=path_by_library,
                    used_paths=used_paths,
                )
                _write_spooled_record(
                    handle=handle,
                    library=library,
                    source_path=source_path,
                    logical_path=current_source.get("logical_path"),
                    relative_path=relative_path,
                    docstring=embedding_text,
                    docstring_raw=docstring,
                    statement=statement,
                    uid=_to_str(toc_row.get("data_uid")),
                    kind=_to_str(toc_row.get("kind")),
                    name=_to_str(toc_row.get("name")),
                    localization=localization,
                )
                counts[library] += 1
                total += 1
                if toc_idx % 10000 == 0:
                    print(f"[scan] toc_rows={toc_idx} docstrings={total}")
    finally:
        for handle in handles.values():
            handle.close()

    info = {
        "table_link_mode": link_mode,
        "num_sources": len(source_list),
        "num_toc_rows": toc_count,
        "source_segment_advances": source_advance_events,
        "source_overflow": source_overflow,
    }
    if link_mode == "by_preorder_segments":
        info["warning"] = (
            "TOC rows had no source_id; source mapping used preorder segment boundaries. "
            "If some sources have zero TOC rows, path/logical-path mapping can drift."
        )

    return path_by_library, dict(counts), total, info


def _load_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _embed_records_for_library(
    records_path: Path,
    expected_count: int,
    batch_size: int,
    embedder: HFEmbedder,
) -> Dict[str, Any]:
    import torch  # type: ignore

    docstrings: List[str] = []
    docstrings_raw: List[str] = []
    statements: List[str] = []
    logical_paths: List[Optional[str]] = []
    relative_paths: List[str] = []
    source_paths: List[str] = []
    localizations: List[Dict[str, Optional[int]]] = []
    uids: List[Optional[str]] = []
    kinds: List[Optional[str]] = []
    names: List[Optional[str]] = []
    chunks: List[Any] = []

    batch_texts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []
    seen = 0

    for row in _load_records(records_path):
        text = _non_empty_str(row.get("docstring"))
        if text is None:
            continue
        batch_texts.append(text)
        batch_meta.append(row)
        seen += 1

        if len(batch_texts) >= batch_size:
            chunks.append(embedder.encode(batch_texts))
            for meta, doc in zip(batch_meta, batch_texts):
                docstrings.append(doc)
                docstrings_raw.append(_to_str(meta.get("docstring_raw")) or "")
                statements.append(_to_str(meta.get("statement")) or "")
                logical_paths.append(_to_str(meta.get("logical_path")))
                relative_paths.append(_to_str(meta.get("relative_path")) or "")
                source_paths.append(_to_str(meta.get("source_path")) or "")
                loc = meta.get("localization")
                localizations.append(loc if isinstance(loc, dict) else {})
                uids.append(_to_str(meta.get("uid")))
                kinds.append(_to_str(meta.get("kind")))
                names.append(_to_str(meta.get("name")))
            batch_texts = []
            batch_meta = []
            if seen % 1000 == 0:
                print(f"[embed] {records_path.name}: {seen}/{expected_count}")

    if batch_texts:
        chunks.append(embedder.encode(batch_texts))
        for meta, doc in zip(batch_meta, batch_texts):
            docstrings.append(doc)
            docstrings_raw.append(_to_str(meta.get("docstring_raw")) or "")
            statements.append(_to_str(meta.get("statement")) or "")
            logical_paths.append(_to_str(meta.get("logical_path")))
            relative_paths.append(_to_str(meta.get("relative_path")) or "")
            source_paths.append(_to_str(meta.get("source_path")) or "")
            loc = meta.get("localization")
            localizations.append(loc if isinstance(loc, dict) else {})
            uids.append(_to_str(meta.get("uid")))
            kinds.append(_to_str(meta.get("kind")))
            names.append(_to_str(meta.get("name")))

    embeddings = torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0), dtype=torch.float32)
    if embeddings.shape[0] != len(docstrings):
        raise RuntimeError(
            f"Embedding/docstring mismatch for {records_path}: "
            f"embeddings={embeddings.shape[0]} docstrings={len(docstrings)}"
        )

    return {
        "format_version": 2,
        "docstrings": docstrings,
        "docstrings_raw": docstrings_raw,
        "statements": statements,
        "logical_paths": logical_paths,
        "relative_paths": relative_paths,
        "source_paths": source_paths,
        "localizations": localizations,
        "uids": uids,
        "kinds": kinds,
        "names": names,
        "embeddings": embeddings,
    }


def _process_env(
    env: str,
    args: argparse.Namespace,
    embedder: HFEmbedder,
) -> Dict[str, Any]:
    output_env_dir = args.output_dir / env
    output_env_dir.mkdir(parents=True, exist_ok=True)

    temp_parent = output_env_dir / "_tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_dir_path = Path(tempfile.mkdtemp(prefix=f"{env}_", dir=str(temp_parent)))

    try:
        print(f"[info] loading HF dataset={args.hf_dataset} env={env}")
        selected_config, row_mode, selected_input = _select_hf_rows(
            dataset_id=args.hf_dataset,
            env=env,
            config_override=args.hf_config,
            split=args.hf_split,
            token=args.hf_token,
            revision=args.hf_revision,
            streaming=bool(args.streaming),
        )
        print(f"[info] selected config={selected_config} mode={row_mode}")

        table_info: Optional[Dict[str, Any]] = None
        if row_mode == "table_only":
            if not isinstance(selected_input, Mapping):
                raise RuntimeError("Internal error: table_only mode requires sources/toc_nodes iterables.")
            records_by_library, counts_by_library, total_docstrings, table_info = (
                _spool_docstrings_by_library_from_tables(
                    sources_rows=selected_input.get("sources", []),
                    toc_rows=selected_input.get("toc_nodes", []),
                    temp_dir=temp_dir_path,
                )
            )
        else:
            records_by_library, counts_by_library, total_docstrings = _spool_docstrings_by_library(
                rows=selected_input,
                row_mode=row_mode,
                temp_dir=temp_dir_path,
            )
        print(
            f"[info] discovered {total_docstrings} docstrings across {len(records_by_library)} libraries"
        )
        if total_docstrings == 0:
            raise RuntimeError("No docstrings found for this env in the selected HF config.")

        summary: Dict[str, Any] = {
            "env": env,
            "hf_dataset": args.hf_dataset,
            "hf_config": selected_config,
            "hf_split": args.hf_split,
            "hf_revision": args.hf_revision,
            "row_mode": row_mode,
            "format_version": 2,
            "streaming": bool(args.streaming),
            "model_name": args.model_name,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "normalize": not args.no_normalize,
            "torch_dtype": args.torch_dtype,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_docstrings": total_docstrings,
            "libraries": [],
        }
        if table_info is not None:
            summary["table_info"] = table_info

        used_output_paths: set[Path] = set()
        for library in sorted(records_by_library):
            records_path = records_by_library[library]
            expected_count = counts_by_library.get(library, 0)
            print(f"[library] {library} ({expected_count} docstrings)")

            payload = _embed_records_for_library(
                records_path=records_path,
                expected_count=expected_count,
                batch_size=args.batch_size,
                embedder=embedder,
            )
            payload.update(
                {
                    "env": env,
                    "library": library,
                    "hf_dataset": args.hf_dataset,
                    "hf_config": selected_config,
                    "hf_split": args.hf_split,
                    "hf_revision": args.hf_revision,
                    "model_name": args.model_name,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

            stem = _safe_filename(library)
            output_file = output_env_dir / f"{stem}.pt"
            if output_file in used_output_paths:
                idx = 1
                while True:
                    candidate = output_env_dir / f"{stem}_{idx}.pt"
                    if candidate not in used_output_paths:
                        output_file = candidate
                        break
                    idx += 1
            used_output_paths.add(output_file)

            import torch  # type: ignore

            torch.save(payload, output_file)
            print(f"[library] wrote {output_file}")

            summary["libraries"].append(
                {
                    "library": library,
                    "docstrings": expected_count,
                    "output_file": str(output_file),
                }
            )

        summary_path = output_env_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        print(f"[done] wrote {summary_path}")
        return summary
    finally:
        if args.keep_temp:
            print(f"[info] keeping temporary files in {temp_dir_path}")
        else:
            shutil.rmtree(temp_dir_path, ignore_errors=True)
            if temp_parent.exists() and not any(temp_parent.iterdir()):
                temp_parent.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-library docstring embeddings using only HF dataset download."
    )
    env_group = parser.add_mutually_exclusive_group(required=True)
    env_group.add_argument("--env", help="Environment id, for example: coq-actuary.")
    env_group.add_argument(
        "--all-envs",
        action="store_true",
        help="Run on all environments discovered from HF dataset configs.",
    )
    parser.add_argument(
        "--hf-dataset",
        default="theostos/pile-of-rocq",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--hf-config",
        default=None,
        help=(
            "Optional HF config override. If omitted, script tries "
            "`<env>-source_blobs`, `<env>`, then `<env>-sources` (+ `<env>-toc_nodes`)."
        ),
    )
    parser.add_argument("--hf-split", default="train", help="HF split name.")
    parser.add_argument("--hf-revision", default=None, help="Optional HF revision (branch/tag/commit).")
    parser.add_argument("--hf-token", default=None, help="Optional HF token.")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use streaming download mode from HF.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-Embedding-4B",
        help="Hugging Face embedding model name.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="Torch device, for example cpu, cuda:0, cuda:1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docstring_embeddings"),
        help="Output root directory.",
    )
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer max length.")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom model code from HF repositories.",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 embedding normalization.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary spool files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be >= 1")
    if args.all_envs and args.hf_config is not None:
        raise ValueError("--hf-config cannot be used with --all-envs.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_envs:
        envs = _discover_envs(
            dataset_id=args.hf_dataset,
            token=args.hf_token,
            revision=args.hf_revision,
        )
        print(f"[info] discovered {len(envs)} envs from dataset configs")
    else:
        assert isinstance(args.env, str)
        envs = [args.env]

    embedder = HFEmbedder(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        trust_remote_code=bool(args.trust_remote_code),
        normalize=not args.no_normalize,
        torch_dtype=args.torch_dtype,
    )
    print(f"[info] model loaded on {args.device}: {args.model_name}")

    all_summaries: List[Dict[str, Any]] = []
    for idx, env in enumerate(envs, start=1):
        print(f"[env {idx}/{len(envs)}] {env}")
        env_summary = _process_env(env=env, args=args, embedder=embedder)
        all_summaries.append(env_summary)

    if args.all_envs:
        run_summary = {
            "hf_dataset": args.hf_dataset,
            "hf_split": args.hf_split,
            "hf_revision": args.hf_revision,
            "streaming": bool(args.streaming),
            "model_name": args.model_name,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "normalize": not args.no_normalize,
            "torch_dtype": args.torch_dtype,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "env_count": len(all_summaries),
            "environments": [
                {
                    "env": summary.get("env"),
                    "hf_config": summary.get("hf_config"),
                    "total_docstrings": summary.get("total_docstrings"),
                }
                for summary in all_summaries
            ],
        }
        summary_path = args.output_dir / "summary_all_envs.json"
        summary_path.write_text(
            json.dumps(run_summary, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
