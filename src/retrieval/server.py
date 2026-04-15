#!/usr/bin/env python3
"""FastAPI + FAISS retrieval server for precomputed docstring embeddings.

Input layout (produced by precompute_embeddings.py):
  <embeddings_root>/
    <env_a>/
      <library_1>.pt
      <library_2>.pt
      ...
    <env_b>/
      ...

Each .pt file is expected to contain:
  - format_version: int (must be 2)
  - embeddings: torch.Tensor [N, D]
  - docstrings: list[str]
  - docstrings_raw: list[str]
  - statements: list[str]
  - logical_paths: list[str | None]
  - localizations: list[dict]
  - kinds: list[str | None] (optional but recommended)
  - names: list[str | None] (optional)

The server exposes:
  - GET  /health
  - GET  /envs
  - GET  /get_libraries?env=<env>
  - POST /search

And caches one FAISS index per env on disk so it is not rebuilt every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_PAYLOAD_FORMAT_VERSION = 2


def _default_device() -> str:
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _as_optional_string_list(value: Any, expected_len: int) -> List[Optional[str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [None] * expected_len
    out: List[Optional[str]] = []
    for item in value[:expected_len]:
        out.append(_to_str(item))
    if len(out) < expected_len:
        out.extend([None] * (expected_len - len(out)))
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


def _resolve_torch_dtype(torch_module: Any, dtype: str) -> Optional[Any]:
    if dtype == "auto":
        return None
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    return mapping[dtype]


class QueryEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int,
        trust_remote_code: bool,
        torch_dtype: str,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependencies for query embedding. Install with: pip install torch transformers"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.max_length = max_length

        model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        dtype = _resolve_torch_dtype(torch, torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
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

    def _extract_embeddings(self, outputs: Any, attention_mask: Optional[Any]) -> Any:
        if self.torch.is_tensor(outputs):
            return outputs

        if isinstance(outputs, Mapping):
            for key in ("sentence_embedding", "embeddings", "pooler_output"):
                value = outputs.get(key)
                if self.torch.is_tensor(value):
                    return value
            last_hidden = outputs.get("last_hidden_state")
            if self.torch.is_tensor(last_hidden):
                return self._mean_pool(last_hidden, attention_mask)

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

        raise RuntimeError("Could not extract embeddings from model output.")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        attention_mask = encoded.get("attention_mask")

        with self.torch.inference_mode():
            outputs = self.model(**encoded)

        embeddings = self._extract_embeddings(outputs, attention_mask)
        if embeddings.dim() == 3:
            embeddings = self._mean_pool(embeddings, attention_mask)
        if embeddings.dim() != 2:
            raise RuntimeError(f"Expected [N, D] embeddings, got shape={tuple(embeddings.shape)}")

        embeddings = embeddings.to(device="cpu", dtype=self.torch.float32)
        arr = embeddings.numpy()
        if arr.size > 0:
            # Cosine retrieval: normalize vectors before search.
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            arr = arr / norms
        return arr.astype(np.float32, copy=False)


class EmbedderPool:
    def __init__(self, device: str, max_length: int, trust_remote_code: bool, torch_dtype: str) -> None:
        self.device = device
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self._models: Dict[str, QueryEmbedder] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str) -> QueryEmbedder:
        with self._lock:
            model = self._models.get(model_name)
            if model is not None:
                return model
            model = QueryEmbedder(
                model_name=model_name,
                device=self.device,
                max_length=self.max_length,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
            )
            self._models[model_name] = model
            return model


@dataclass
class EnvRuntime:
    env: str
    index: Any
    model_name: str
    docstrings: List[str]
    docstrings_raw: List[str]
    statements: List[str]
    logical_paths: List[Optional[str]]
    localizations: List[Dict[str, Any]]
    kinds: List[Optional[str]]
    names: List[Optional[str]]
    relative_paths: List[str]
    libraries: List[str]
    fingerprint: str

    @property
    def size(self) -> int:
        return len(self.docstrings)

    @property
    def available_libraries(self) -> List[str]:
        return sorted({lib for lib in self.libraries if isinstance(lib, str) and lib})

    @property
    def available_kinds(self) -> List[str]:
        return sorted({kind for kind in self.kinds if isinstance(kind, str) and kind})


class RetrievalService:
    def __init__(
        self,
        embeddings_root: Path,
        cache_dir: Path,
        default_query_model: Optional[str],
        device: str,
        max_length: int,
        trust_remote_code: bool,
        torch_dtype: str,
    ) -> None:
        self.embeddings_root = embeddings_root
        self.cache_dir = cache_dir
        self.default_query_model = default_query_model
        self.embedder_pool = EmbedderPool(
            device=device,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_by_env: Dict[str, EnvRuntime] = {}
        self._locks_by_env: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def list_envs(self) -> List[str]:
        if not self.embeddings_root.exists():
            return []
        cache_dir_resolved = self.cache_dir.resolve()
        envs: List[str] = []
        for path in sorted(self.embeddings_root.iterdir()):
            if not path.is_dir():
                continue
            if path.name.startswith("."):
                continue
            try:
                if path.resolve() == cache_dir_resolved:
                    continue
            except OSError:
                pass
            if any(path.glob("*.pt")):
                envs.append(path.name)
        return envs

    def list_libraries(self, env: str) -> List[str]:
        if not isinstance(env, str) or not env.strip():
            raise ValueError("env must be a non-empty string.")
        runtime = self._get_env_runtime(env.strip())
        return runtime.available_libraries

    def _normalize_filter_values(self, values: Optional[Sequence[str]]) -> Optional[List[str]]:
        if values is None:
            return None
        out: List[str] = []
        seen: set[str] = set()
        for value in values:
            text = _to_str(value)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
        if not out:
            return None
        return out

    def _build_allowed_mask(
        self,
        runtime: EnvRuntime,
        *,
        libraries: Optional[Sequence[str]],
        kinds: Optional[Sequence[str]],
    ) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[str]]]:
        selected_libraries = self._normalize_filter_values(libraries)
        selected_kinds = self._normalize_filter_values(kinds)

        if selected_libraries is not None:
            available = set(runtime.available_libraries)
            unknown = [lib for lib in selected_libraries if lib not in available]
            if unknown:
                raise ValueError(
                    "Unknown libraries for env "
                    f"{runtime.env}: {unknown}. Available: {runtime.available_libraries}"
                )

        if selected_kinds is not None:
            available = set(runtime.available_kinds)
            unknown = [kind for kind in selected_kinds if kind not in available]
            if unknown:
                raise ValueError(
                    "Unknown kinds for env "
                    f"{runtime.env}: {unknown}. Available: {runtime.available_kinds}"
                )

        allowed = np.ones(runtime.size, dtype=bool)

        if selected_libraries is not None:
            selected_set = set(selected_libraries)
            lib_mask = np.fromiter((lib in selected_set for lib in runtime.libraries), dtype=bool, count=runtime.size)
            allowed &= lib_mask

        if selected_kinds is not None:
            selected_set = set(selected_kinds)
            kind_mask = np.fromiter(
                ((isinstance(kind, str) and kind in selected_set) for kind in runtime.kinds),
                dtype=bool,
                count=runtime.size,
            )
            allowed &= kind_mask

        return allowed, selected_libraries, selected_kinds

    def search(
        self,
        query: str,
        env: str,
        k: int,
        libraries: Optional[Sequence[str]] = None,
        kinds: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(env, str) or not env.strip():
            raise ValueError("env must be a non-empty string.")
        if k < 1:
            raise ValueError("k must be >= 1.")

        runtime = self._get_env_runtime(env.strip())
        allowed_mask, selected_libraries, selected_kinds = self._build_allowed_mask(
            runtime,
            libraries=libraries,
            kinds=kinds,
        )
        candidate_count = int(allowed_mask.sum())

        model = self.embedder_pool.get(runtime.model_name)
        qvec = model.encode([query.strip()])

        if qvec.shape[1] != runtime.index.d:
            raise RuntimeError(
                f"Query embedding dim ({qvec.shape[1]}) does not match index dim ({runtime.index.d})."
            )

        kk = min(k, candidate_count)
        if kk == 0:
            return {
                "env": runtime.env,
                "query": query,
                "k": k,
                "returned": 0,
                "model_name": runtime.model_name,
                "index_size": runtime.size,
                "candidate_count": candidate_count,
                "libraries": selected_libraries if selected_libraries is not None else runtime.available_libraries,
                "kinds": selected_kinds if selected_kinds is not None else runtime.available_kinds,
                "results": [],
            }

        hits: List[Dict[str, Any]] = []
        search_k = min(max(kk * 4, 32), runtime.size)

        while True:
            scores, ids = runtime.index.search(qvec, search_k)
            for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
                if idx < 0:
                    continue
                if not allowed_mask[idx]:
                    continue
                hits.append(
                    {
                        "rank": len(hits) + 1,
                        "score": float(score),
                        "docstring": runtime.docstrings[idx],
                        "docstring_raw": runtime.docstrings_raw[idx],
                        "statement": runtime.statements[idx],
                        "logical_path": runtime.logical_paths[idx],
                        "localization": runtime.localizations[idx],
                        "relative_path": runtime.relative_paths[idx],
                        "library": runtime.libraries[idx],
                        "kind": runtime.kinds[idx],
                        "name": runtime.names[idx],
                    }
                )
                if len(hits) >= kk:
                    break

            if len(hits) >= kk or search_k >= runtime.size:
                break
            search_k = min(runtime.size, search_k * 2)

        return {
            "env": runtime.env,
            "query": query,
            "k": k,
            "returned": len(hits),
            "model_name": runtime.model_name,
            "index_size": runtime.size,
            "candidate_count": candidate_count,
            "libraries": selected_libraries if selected_libraries is not None else runtime.available_libraries,
            "kinds": selected_kinds if selected_kinds is not None else runtime.available_kinds,
            "results": hits,
        }

    def _get_env_runtime(self, env: str) -> EnvRuntime:
        with self._global_lock:
            cached = self._runtime_by_env.get(env)
            if cached is not None:
                return cached
            lock = self._locks_by_env.get(env)
            if lock is None:
                lock = threading.Lock()
                self._locks_by_env[env] = lock

        with lock:
            with self._global_lock:
                cached = self._runtime_by_env.get(env)
                if cached is not None:
                    return cached

            runtime = self._load_or_build_env(env)
            with self._global_lock:
                self._runtime_by_env[env] = runtime
            return runtime

    def _env_pt_files(self, env: str) -> List[Path]:
        env_dir = self.embeddings_root / env
        if not env_dir.exists() or not env_dir.is_dir():
            raise FileNotFoundError(f"Unknown environment: {env}")
        try:
            if env_dir.resolve() == self.cache_dir.resolve():
                raise FileNotFoundError(f"Unknown environment: {env}")
        except OSError:
            pass
        files = sorted(p for p in env_dir.glob("*.pt") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No .pt embedding files found for env: {env}")
        return files

    def _env_fingerprint(self, env: str, pt_files: Sequence[Path]) -> str:
        hasher = hashlib.sha256()
        hasher.update(env.encode("utf-8"))
        for path in pt_files:
            st = path.stat()
            hasher.update(path.name.encode("utf-8"))
            hasher.update(str(st.st_size).encode("utf-8"))
            hasher.update(str(st.st_mtime_ns).encode("utf-8"))
        return hasher.hexdigest()[:16]

    def _cache_paths(self, env: str, fingerprint: str) -> Tuple[Path, Path]:
        safe_env = env.replace("/", "_")
        base = self.cache_dir / f"{safe_env}__{fingerprint}"
        return base.with_suffix(".faiss"), base.with_suffix(".meta.pt")

    def _load_or_build_env(self, env: str) -> EnvRuntime:
        pt_files = self._env_pt_files(env)
        fingerprint = self._env_fingerprint(env, pt_files)
        index_path, meta_path = self._cache_paths(env, fingerprint)

        if index_path.exists() and meta_path.exists():
            try:
                return self._load_cached_runtime(env, fingerprint, index_path, meta_path)
            except RuntimeError as exc:
                print(f"[cache] rebuilding env={env} after cache load failure: {exc}")

        runtime = self._build_runtime(env, fingerprint, pt_files)
        self._save_runtime_cache(runtime, index_path, meta_path)
        return runtime

    def _load_cached_runtime(
        self,
        env: str,
        fingerprint: str,
        index_path: Path,
        meta_path: Path,
    ) -> EnvRuntime:
        try:
            import faiss  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Missing dependencies. Install with: pip install faiss-cpu torch") from exc

        index = faiss.read_index(str(index_path))
        meta = torch.load(meta_path, map_location="cpu")
        if not isinstance(meta, Mapping):
            raise RuntimeError(f"Invalid cache metadata format: {meta_path}")
        format_version = _to_int(meta.get("format_version"))
        if format_version != EMBEDDING_PAYLOAD_FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported cache metadata format_version={format_version} for env={env}. "
                "Delete the cache directory and restart the server."
            )

        docstrings = _as_string_list(meta.get("docstrings"), expected_len=int(meta.get("size", 0)), fallback="")
        docstrings_raw = _as_string_list(meta.get("docstrings_raw"), expected_len=len(docstrings), fallback="")
        statements = _as_string_list(meta.get("statements"), expected_len=len(docstrings), fallback="")
        logical_paths = _as_optional_string_list(meta.get("logical_paths"), expected_len=len(docstrings))
        localizations = _as_mapping_list(meta.get("localizations"), expected_len=len(docstrings))
        kinds = _as_optional_string_list(meta.get("kinds"), expected_len=len(docstrings))
        names = _as_optional_string_list(meta.get("names"), expected_len=len(docstrings))
        relative_paths = _as_string_list(meta.get("relative_paths"), expected_len=len(docstrings), fallback="")
        libraries = _as_string_list(meta.get("libraries"), expected_len=len(docstrings), fallback="")

        if index.ntotal != len(docstrings):
            raise RuntimeError(
                f"Cache mismatch for env={env}: faiss ntotal={index.ntotal} meta rows={len(docstrings)}"
            )

        model_name = _to_str(meta.get("model_name")) or self.default_query_model or DEFAULT_EMBEDDING_MODEL
        return EnvRuntime(
            env=env,
            index=index,
            model_name=model_name,
            docstrings=docstrings,
            docstrings_raw=docstrings_raw,
            statements=statements,
            logical_paths=logical_paths,
            localizations=localizations,
            kinds=kinds,
            names=names,
            relative_paths=relative_paths,
            libraries=libraries,
            fingerprint=fingerprint,
        )

    def _build_runtime(self, env: str, fingerprint: str, pt_files: Sequence[Path]) -> EnvRuntime:
        try:
            import faiss  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Missing dependencies. Install with: pip install faiss-cpu torch") from exc

        vectors: List[np.ndarray] = []
        docstrings: List[str] = []
        docstrings_raw: List[str] = []
        statements: List[str] = []
        logical_paths: List[Optional[str]] = []
        localizations: List[Dict[str, Any]] = []
        kinds: List[Optional[str]] = []
        names: List[Optional[str]] = []
        relative_paths: List[str] = []
        libraries: List[str] = []
        model_counter: Counter[str] = Counter()

        for path in pt_files:
            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, Mapping):
                continue
            format_version = _to_int(payload.get("format_version"))
            if format_version != EMBEDDING_PAYLOAD_FORMAT_VERSION:
                raise RuntimeError(
                    f"{path}: unsupported embedding payload format_version={format_version}. "
                    "Run src/retrieval/migrate_embeddings_format.py to upgrade existing files."
                )

            emb = payload.get("embeddings")
            if emb is None:
                continue
            if hasattr(emb, "detach"):
                arr = emb.detach().cpu().numpy()
            else:
                arr = np.asarray(emb)
            if arr.ndim != 2:
                raise RuntimeError(f"{path}: embeddings must be 2D, got shape={arr.shape}")
            arr = arr.astype(np.float32, copy=False)
            n = int(arr.shape[0])
            if n <= 0:
                continue

            ds = _as_string_list(payload.get("docstrings"), expected_len=n, fallback="")
            dr = _as_string_list(payload.get("docstrings_raw"), expected_len=n, fallback="")
            st = _as_string_list(payload.get("statements"), expected_len=n, fallback="")
            lp = _as_optional_string_list(payload.get("logical_paths"), expected_len=n)
            loc = _as_mapping_list(payload.get("localizations"), expected_len=n)
            kd = _as_optional_string_list(payload.get("kinds"), expected_len=n)
            nm = _as_optional_string_list(payload.get("names"), expected_len=n)
            rp = _as_string_list(payload.get("relative_paths"), expected_len=n, fallback="")
            lib_field = _to_str(payload.get("library"))
            libs = [lib_field or path.stem] * n

            vectors.append(arr)
            docstrings.extend(ds)
            docstrings_raw.extend(dr)
            statements.extend(st)
            logical_paths.extend(lp)
            localizations.extend(loc)
            kinds.extend(kd)
            names.extend(nm)
            relative_paths.extend(rp)
            libraries.extend(libs)

            model_name = _to_str(payload.get("model_name"))
            if isinstance(model_name, str) and model_name:
                model_counter[model_name] += n

        if not vectors:
            raise RuntimeError(f"No non-empty embedding tensors found for env={env}.")

        matrix = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
        if matrix.shape[0] != len(docstrings):
            raise RuntimeError(
                f"Metadata/embedding mismatch for env={env}: vectors={matrix.shape[0]} docstrings={len(docstrings)}"
            )

        # Cosine similarity retrieval.
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(int(matrix.shape[1]))
        index.add(matrix)

        if self.default_query_model:
            model_name = self.default_query_model
        elif model_counter:
            model_name = model_counter.most_common(1)[0][0]
        else:
            model_name = DEFAULT_EMBEDDING_MODEL

        return EnvRuntime(
            env=env,
            index=index,
            model_name=model_name,
            docstrings=docstrings,
            docstrings_raw=docstrings_raw,
            statements=statements,
            logical_paths=logical_paths,
            localizations=localizations,
            kinds=kinds,
            names=names,
            relative_paths=relative_paths,
            libraries=libraries,
            fingerprint=fingerprint,
        )

    def _save_runtime_cache(self, runtime: EnvRuntime, index_path: Path, meta_path: Path) -> None:
        try:
            import faiss  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Missing dependencies. Install with: pip install faiss-cpu torch") from exc

        index_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(runtime.index, str(index_path))

        meta = {
            "format_version": EMBEDDING_PAYLOAD_FORMAT_VERSION,
            "env": runtime.env,
            "fingerprint": runtime.fingerprint,
            "size": runtime.size,
            "dim": runtime.index.d,
            "model_name": runtime.model_name,
            "docstrings": runtime.docstrings,
            "docstrings_raw": runtime.docstrings_raw,
            "statements": runtime.statements,
            "logical_paths": runtime.logical_paths,
            "localizations": runtime.localizations,
            "kinds": runtime.kinds,
            "names": runtime.names,
            "relative_paths": runtime.relative_paths,
            "libraries": runtime.libraries,
        }
        torch.save(meta, meta_path)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    env: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=1000)
    libraries: Optional[List[str]] = None
    kinds: Optional[List[str]] = None


def create_app(service: RetrievalService) -> FastAPI:
    app = FastAPI(title="Docstring Retrieval Server", version="1.0.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

    @app.get("/envs")
    def list_envs() -> Dict[str, Any]:
        return {"environments": service.list_envs()}

    @app.get("/get_libraries")
    def get_libraries(env: str) -> Dict[str, Any]:
        try:
            return {"env": env, "libraries": service.list_libraries(env)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/search")
    def search(req: SearchRequest) -> Dict[str, Any]:
        try:
            return service.search(
                query=req.query,
                env=req.env,
                k=req.k,
                libraries=req.libraries,
                kinds=req.kinds,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve docstring retrieval over precomputed embeddings.")
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=Path("docstring_embeddings"),
        help="Root directory containing one subdirectory per environment.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="FAISS cache directory. Default: <embeddings-root>/.faiss_cache",
    )
    parser.add_argument(
        "--default-query-model",
        default=None,
        help=(
            "Optional override model for query embedding. "
            "If omitted, the model_name stored in .pt files is used."
        ),
    )
    parser.add_argument("--device", default=_default_device(), help="Torch device for query embedding.")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer max length for query embedding.")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype for query embedding model.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom HF model code.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="FastAPI bind host.")
    parser.add_argument("--port", type=int, default=8010, help="FastAPI bind port.")
    parser.add_argument(
        "--preload-all",
        action="store_true",
        help="Build/load all env indexes at startup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.embeddings_root.exists():
        raise FileNotFoundError(f"Embeddings root not found: {args.embeddings_root}")
    if not args.embeddings_root.is_dir():
        raise NotADirectoryError(f"Embeddings root is not a directory: {args.embeddings_root}")
    if args.max_length < 1:
        raise ValueError("--max-length must be >= 1")

    cache_dir = args.cache_dir if args.cache_dir is not None else (args.embeddings_root / ".faiss_cache")
    service = RetrievalService(
        embeddings_root=args.embeddings_root,
        cache_dir=cache_dir,
        default_query_model=args.default_query_model,
        device=args.device,
        max_length=args.max_length,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=args.torch_dtype,
    )

    if args.preload_all:
        envs = service.list_envs()
        for env in envs:
            print(f"[preload] {env}")
            service._get_env_runtime(env)  # noqa: SLF001

    app = create_app(service)

    try:
        import uvicorn  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency. Install with: pip install uvicorn") from exc

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
