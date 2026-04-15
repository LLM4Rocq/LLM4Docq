#!/usr/bin/env python3
"""Temporary one-off migration tool for retrieval embedding payloads.

Upgrades existing per-library `.pt` files to format v2 without recomputing
embeddings. It enriches metadata by:
- deriving line/character localization from byte offsets + source UTF-8 content
- extracting full statements from source bytes
- rewriting `docstrings` to include full statement + original docstring

Source content is loaded from Hugging Face `<env>-sources`.
"""

from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

FORMAT_VERSION = 2


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


def _as_string_list(value: Any, expected_len: int, fallback: str = "") -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [fallback] * expected_len
    out: List[str] = []
    for item in value[:expected_len]:
        s = _to_str(item)
        out.append(s if isinstance(s, str) else fallback)
    if len(out) < expected_len:
        out.extend([fallback] * (expected_len - len(out)))
    return out


def _as_mapping_list(value: Any, expected_len: int) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [{} for _ in range(expected_len)]
    out: List[Dict[str, Any]] = []
    for item in value[:expected_len]:
        out.append(dict(item) if isinstance(item, Mapping) else {})
    if len(out) < expected_len:
        out.extend({} for _ in range(expected_len - len(out)))
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

    span = row.get("span")
    if (start_bp is None or end_bp is None) and isinstance(span, Mapping):
        if start_bp is None:
            start_bp = _to_int(span.get("start_bp"))
        if end_bp is None:
            end_bp = _to_int(span.get("end_bp"))
        if start_bp is None:
            start_bp = _to_int(span.get("bp"))
        if end_bp is None:
            end_bp = _to_int(span.get("ep"))

    if start_bp is None:
        start_bp = _to_int(row.get("bp"))
    if end_bp is None:
        end_bp = _to_int(row.get("ep"))
    return _normalize_bp_ep(start_bp, end_bp)


@dataclass
class SourceTextView:
    content_bytes: bytes
    line_starts: List[int]



def _build_line_starts(content_bytes: bytes) -> List[int]:
    starts = [0]
    cursor = 0
    while True:
        idx = content_bytes.find(b"\n", cursor)
        if idx < 0:
            break
        starts.append(idx + 1)
        cursor = idx + 1
    return starts


def _build_source_view(content: str) -> SourceTextView:
    content_bytes = content.encode("utf-8")
    return SourceTextView(content_bytes=content_bytes, line_starts=_build_line_starts(content_bytes))


def _offset_to_line_char(view: SourceTextView, offset: int) -> Tuple[int, int]:
    offset = max(0, min(offset, len(view.content_bytes)))
    line = bisect.bisect_right(view.line_starts, offset) - 1
    if line < 0:
        line = 0
    line_start = view.line_starts[line] if line < len(view.line_starts) else 0
    segment = view.content_bytes[line_start:offset]
    character = len(segment.decode("utf-8", errors="ignore"))
    return line, character


def _localization_from_offsets(
    view: Optional[SourceTextView],
    bp: Optional[int],
    ep: Optional[int],
    old_loc: Mapping[str, Any],
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

    if view is not None and (bp is not None or ep is not None):
        if bp is not None:
            sl, sc = _offset_to_line_char(view, bp)
            out["start_line"] = sl
            out["start_character"] = sc
        if ep is not None:
            el, ec = _offset_to_line_char(view, ep)
            out["end_line"] = el
            out["end_character"] = ec

    if out["start_line"] is None:
        out["start_line"] = _to_int(old_loc.get("start_line"))
    if out["start_character"] is None:
        out["start_character"] = _to_int(old_loc.get("start_character"))
    if out["end_line"] is None:
        out["end_line"] = _to_int(old_loc.get("end_line"))
    if out["end_character"] is None:
        out["end_character"] = _to_int(old_loc.get("end_character"))
    return out


def _statement_from_offsets(view: Optional[SourceTextView], bp: Optional[int], ep: Optional[int]) -> str:
    bp, ep = _normalize_bp_ep(bp, ep)
    if view is None or bp is None or ep is None:
        return ""
    bp = max(0, min(bp, len(view.content_bytes)))
    ep = max(bp, min(ep, len(view.content_bytes)))
    if ep <= bp:
        return ""
    return view.content_bytes[bp:ep].decode("utf-8", errors="replace").strip()


def _merge_statement_and_docstring(statement: str, docstring: str) -> str:
    statement = statement.strip()
    docstring = docstring.strip()
    if statement and docstring:
        return f"{statement}\n\n{docstring}"
    if statement:
        return statement
    return docstring


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


def _discover_local_envs(input_dir: Path) -> List[str]:
    envs: List[str] = []
    if not input_dir.exists():
        return envs
    for path in sorted(input_dir.iterdir()):
        if not path.is_dir():
            continue
        if any(path.glob("*.pt")):
            envs.append(path.name)
    return envs


def _embedding_row_count(payload: Mapping[str, Any]) -> int:
    emb = payload.get("embeddings")
    shape = getattr(emb, "shape", None)
    if shape is not None and len(shape) >= 1:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            pass
    docs = payload.get("docstrings")
    if isinstance(docs, Sequence) and not isinstance(docs, (str, bytes, bytearray)):
        return len(docs)
    return 0


def _collect_required_keys(pt_files: Sequence[Path], torch_module: Any) -> Dict[str, set[Tuple[str, str]]]:
    required: Dict[str, set[Tuple[str, str]]] = {}
    for path in pt_files:
        payload = torch_module.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            continue
        n = _embedding_row_count(payload)
        if n <= 0:
            continue
        library = _to_str(payload.get("library")) or path.stem
        rel_paths = _as_string_list(payload.get("relative_paths"), expected_len=n, fallback="")
        src_paths = _as_string_list(payload.get("source_paths"), expected_len=n, fallback="")
        keys: set[Tuple[str, str]] = set()
        for rel_path, src_path in zip(rel_paths, src_paths):
            rel = rel_path.strip()
            if not rel and src_path:
                rel = _source_relative_path(src_path, library)
            if rel:
                keys.add((library, rel))
        required[path.name] = keys
    return required


def _load_source_views_for_env(
    dataset_id: str,
    env: str,
    split: str,
    token: Optional[str],
    revision: Optional[str],
    streaming: bool,
    required_keys: set[Tuple[str, str]],
) -> Dict[Tuple[str, str], SourceTextView]:
    config_name = f"{env}-sources"
    dataset = _load_dataset_once(
        dataset_id=dataset_id,
        config_name=config_name,
        split=split,
        token=token,
        revision=revision,
        streaming=streaming,
    )

    out: Dict[Tuple[str, str], SourceTextView] = {}
    for row in dataset:
        if not isinstance(row, Mapping):
            continue
        library = _to_str(row.get("library"))
        source_path = _to_str(row.get("source_path"))
        content = _to_str(row.get("content"))
        if not isinstance(library, str) or not isinstance(source_path, str) or not isinstance(content, str):
            continue
        rel = _source_relative_path(source_path, library)
        key = (library, rel)
        if required_keys and key not in required_keys:
            continue
        if key not in out:
            out[key] = _build_source_view(content)
        if required_keys and len(out) >= len(required_keys):
            break
    return out


def _migrate_payload(
    payload: Mapping[str, Any],
    path: Path,
    source_views: Mapping[Tuple[str, str], SourceTextView],
) -> Dict[str, Any]:
    n = _embedding_row_count(payload)
    if n <= 0:
        out = dict(payload)
        out["format_version"] = FORMAT_VERSION
        out.setdefault("docstrings_raw", [])
        out.setdefault("statements", [])
        out.setdefault("localizations", [])
        out["migrated_at_utc"] = datetime.now(timezone.utc).isoformat()
        out["migrated_from_format_version"] = _to_int(payload.get("format_version")) or 1
        return out

    library = _to_str(payload.get("library")) or path.stem

    docs_current = _as_string_list(payload.get("docstrings"), expected_len=n, fallback="")
    docs_raw_in = _as_string_list(payload.get("docstrings_raw"), expected_len=n, fallback="")
    statements_in = _as_string_list(payload.get("statements"), expected_len=n, fallback="")
    rel_paths = _as_string_list(payload.get("relative_paths"), expected_len=n, fallback="")
    src_paths = _as_string_list(payload.get("source_paths"), expected_len=n, fallback="")
    old_locs = _as_mapping_list(payload.get("localizations"), expected_len=n)

    docs_new: List[str] = []
    docs_raw_new: List[str] = []
    statements_new: List[str] = []
    locs_new: List[Dict[str, Optional[int]]] = []

    for idx in range(n):
        old_doc = docs_current[idx]
        raw_doc = docs_raw_in[idx].strip() or old_doc.strip()

        rel = rel_paths[idx].strip()
        if not rel and src_paths[idx]:
            rel = _source_relative_path(src_paths[idx], library)

        source_view = source_views.get((library, rel)) if rel else None
        old_loc = old_locs[idx]
        bp, ep = _extract_bp_ep(old_loc)

        statement = _statement_from_offsets(source_view, bp, ep)
        if not statement and statements_in[idx].strip():
            statement = statements_in[idx].strip()

        merged_doc = _merge_statement_and_docstring(statement, raw_doc)
        if not merged_doc:
            merged_doc = old_doc

        loc_new = _localization_from_offsets(source_view, bp, ep, old_loc)

        docs_new.append(merged_doc)
        docs_raw_new.append(raw_doc)
        statements_new.append(statement)
        locs_new.append(loc_new)

    out = dict(payload)
    out["format_version"] = FORMAT_VERSION
    out["docstrings"] = docs_new
    out["docstrings_raw"] = docs_raw_new
    out["statements"] = statements_new
    out["localizations"] = locs_new
    out["migrated_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["migrated_from_format_version"] = _to_int(payload.get("format_version")) or 1
    return out


def _save_payload(torch_module: Any, payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch_module.save(dict(payload), tmp_path)
    tmp_path.replace(output_path)


def _migrate_env(
    env: str,
    input_dir: Path,
    output_dir: Path,
    dataset_id: str,
    split: str,
    token: Optional[str],
    revision: Optional[str],
    streaming: bool,
    overwrite: bool,
    torch_module: Any,
) -> Dict[str, Any]:
    input_env_dir = input_dir / env
    output_env_dir = output_dir / env

    pt_files = sorted(p for p in input_env_dir.glob("*.pt") if p.is_file())
    if not pt_files:
        raise RuntimeError(f"[{env}] no .pt files found in {input_env_dir}")

    required_by_file = _collect_required_keys(pt_files, torch_module)
    required_all: set[Tuple[str, str]] = set()
    for keys in required_by_file.values():
        required_all.update(keys)

    print(f"[env] {env}: loading source contents from HF ({len(required_all)} required paths)")
    source_views = _load_source_views_for_env(
        dataset_id=dataset_id,
        env=env,
        split=split,
        token=token,
        revision=revision,
        streaming=streaming,
        required_keys=required_all,
    )
    missing_count = max(0, len(required_all) - len(source_views))
    if missing_count > 0:
        print(f"[env] {env}: warning missing {missing_count} source contents, fallback metadata will be used")

    output_env_dir.mkdir(parents=True, exist_ok=True)

    migrated_files = 0
    for in_path in pt_files:
        out_path = output_env_dir / in_path.name
        if out_path.exists() and not overwrite and out_path.resolve() != in_path.resolve():
            raise FileExistsError(f"Output file already exists: {out_path} (use --overwrite)")

        payload = torch_module.load(in_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Invalid payload in {in_path}: expected a mapping")

        migrated = _migrate_payload(payload, in_path, source_views)
        _save_payload(torch_module, migrated, out_path)
        migrated_files += 1
        print(f"[env] {env}: wrote {out_path}")

    in_summary = input_env_dir / "summary.json"
    out_summary = output_env_dir / "summary.json"
    if in_summary.exists():
        try:
            data = json.loads(in_summary.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["format_version"] = FORMAT_VERSION
                data["migration"] = {
                    "tool": "migrate_embeddings_format.py",
                    "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "migrated_files": migrated_files,
                    "hf_dataset": dataset_id,
                    "hf_split": split,
                    "hf_revision": revision,
                }
                out_summary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[env] {env}: warning could not migrate summary.json: {exc}")

    return {
        "env": env,
        "files": migrated_files,
        "required_sources": len(required_all),
        "loaded_sources": len(source_views),
        "missing_sources": missing_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporary migration: upgrade retrieval embedding payloads to format v2 without recomputing embeddings."
    )
    env_group = parser.add_mutually_exclusive_group(required=True)
    env_group.add_argument("--env", help="Single environment to migrate.")
    env_group.add_argument("--all-envs", action="store_true", help="Migrate all envs found under --input-dir.")

    parser.add_argument("--input-dir", type=Path, default=Path("envs"), help="Input root containing <env>/*.pt files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root. Default: <input-dir>_v2 (unless --in-place).",
    )
    parser.add_argument("--in-place", action="store_true", help="Rewrite files directly in --input-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in --output-dir.")

    parser.add_argument("--hf-dataset", default="theostos/pile-of-rocq", help="HF dataset id.")
    parser.add_argument("--hf-split", default="train", help="HF split name.")
    parser.add_argument("--hf-revision", default=None, help="Optional HF dataset revision.")
    parser.add_argument("--hf-token", default=None, help="Optional HF token.")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HF streaming mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input dir is not a directory: {args.input_dir}")

    if args.in_place and args.output_dir is not None:
        raise ValueError("--output-dir cannot be used with --in-place")

    if args.in_place:
        output_dir = args.input_dir
    else:
        output_dir = args.output_dir if args.output_dir is not None else Path(str(args.input_dir) + "_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    envs: List[str]
    if args.all_envs:
        envs = _discover_local_envs(args.input_dir)
        if not envs:
            raise RuntimeError(f"No environments found under {args.input_dir}")
    else:
        assert isinstance(args.env, str)
        envs = [args.env]

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: pip install torch") from exc

    results: List[Dict[str, Any]] = []
    for idx, env in enumerate(envs, start=1):
        print(f"[run] env {idx}/{len(envs)}: {env}")
        result = _migrate_env(
            env=env,
            input_dir=args.input_dir,
            output_dir=output_dir,
            dataset_id=args.hf_dataset,
            split=args.hf_split,
            token=args.hf_token,
            revision=args.hf_revision,
            streaming=bool(args.streaming),
            overwrite=bool(args.overwrite) or bool(args.in_place),
            torch_module=torch,
        )
        results.append(result)

    summary = {
        "tool": "migrate_embeddings_format.py",
        "format_version": FORMAT_VERSION,
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "hf_dataset": args.hf_dataset,
        "hf_split": args.hf_split,
        "hf_revision": args.hf_revision,
        "streaming": bool(args.streaming),
        "env_count": len(results),
        "environments": results,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_dir / "migration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
