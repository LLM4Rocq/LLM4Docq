#!/usr/bin/env python3
"""Patch local OPAM images with env/source TOC+docstrings from HF parquet exports.

For each Pile-of-Rocq config whose `name:tag` image exists locally, this script:
1) loads `<env>/sources.parquet`, `<env>/toc_nodes.parquet`, `<env>/env_toc.parquet`
   from either a local mirror or Hugging Face Hub,
2) reconstructs:
   - env TOC JSON: `<opam_env_path>/lib/coq/<env>.toc.json`
   - per-source TOC JSON (with docstrings): `<source>.v.toc.json`,
3) starts the image through `OpamDocker` (optionally updates rocq-ml-toolbox),
4) optionally installs the retrieval server package in-container,
5) optionally migrates/copies env docstring embeddings in-container,
6) writes TOC files in-container, commits to an output image ref, and optionally pushes.

Default output image ref is the same repository/tag as the input config image.
"""

from __future__ import annotations

import argparse
import io
import json
import shlex
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import docker
import pyarrow.parquet as pq

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None

# Prefer installed package, fallback to sibling checkout.
try:
    from rocq_ml_toolbox.docker.opam_docker import OpamConfig, OpamDocker
except ImportError:  # pragma: no cover
    import sys

    THIS_FILE = Path(__file__).resolve()
    sibling_src = THIS_FILE.parents[2] / "rocq-ml-toolbox" / "src"
    if sibling_src.exists():
        sys.path.insert(0, str(sibling_src))
    from rocq_ml_toolbox.docker.opam_docker import OpamConfig, OpamDocker


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    source_idx: int
    source_path_canonical: str


@dataclass(frozen=True)
class EnvJob:
    env: str
    config_path: Path
    config: OpamConfig


RETRIEVAL_RUNTIME_DEPENDENCIES: Tuple[str, ...] = (
    "numpy>=1.24",
    "pydantic>=2,<3",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "torch>=2.1",
    "transformers>=4.45",
    "faiss-cpu>=1.8.0",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _resolve_path(candidate: Path, *, must_exist: bool) -> Path:
    probe = candidate
    if probe.exists() or not must_exist:
        return probe.resolve()

    fallback = Path("..") / candidate
    if fallback.exists() or not must_exist:
        return fallback.resolve()

    raise FileNotFoundError(f"path does not exist: {candidate}")


def _prepare_targets(opam_docker: OpamDocker, config: OpamConfig) -> None:
    """Mirror step_2_dump._prepare_targets without importing extraction step modules."""
    for target in config.targets:
        target_name = target.lib
        extra_args = getattr(target, "extra_coq_proj_args", None) or []
        opam_docker.add_coqproject(target_name, extra_args)

        if getattr(target, "copy_skeleton", False):
            for package in getattr(target, "packages", None) or []:
                opam_docker.copy_coq_files_from_package_to_target(package, target_name)

        if getattr(target, "copy_elpi", False):
            for package in getattr(target, "packages", None) or []:
                opam_docker.copy_elpi_files_from_package_to_target(package, target_name)


def _put_archive(opam_docker: OpamDocker, dest_dir: str, archive_bytes: bytes) -> None:
    ok = opam_docker.container.put_archive(dest_dir, archive_bytes)
    if not ok:
        raise RuntimeError(f"failed to put archive into container at {dest_dir}")


def _exec_in_container(
    opam_docker: OpamDocker,
    cmd: Sequence[str] | str,
    *,
    user: Optional[str] = None,
    check: bool = True,
) -> str:
    opam_docker._ensure_running()  # noqa: SLF001
    api = opam_docker.client.api
    exec_id = api.exec_create(
        opam_docker.container.id,
        cmd,
        stdout=True,
        stderr=True,
        tty=False,
        user=user or opam_docker.config.user,
    )["Id"]
    stdout, stderr = api.exec_start(exec_id, demux=True)
    status = api.exec_inspect(exec_id)
    code = int(status.get("ExitCode", 1))

    out_text = (stdout or b"").decode("utf-8", errors="replace")
    err_text = (stderr or b"").decode("utf-8", errors="replace")
    if check and code != 0:
        raise RuntimeError(
            f"Command failed in container (exit={code}).\n"
            f"cmd={cmd}\n"
            f"stdout:\n{out_text}\n"
            f"stderr:\n{err_text}"
        )
    return out_text


def _sync_retrieval_project_and_install(
    opam_docker: OpamDocker,
    config: OpamConfig,
    project_root: Path,
    *,
    container_project_dir: Optional[str],
    install_runtime_dependencies: bool,
) -> str:
    home_dir = f"/home/{config.user}"
    target_dir = container_project_dir or f"{home_dir}/LLM4Docq"
    target_parent = str(PurePosixPath(target_dir).parent)
    target_name = PurePosixPath(target_dir).name

    include_paths = [
        Path("pyproject.toml"),
        Path("README.md"),
        Path("src/retrieval"),
    ]

    buf = io.BytesIO()

    def _writable_permissions(info: tarfile.TarInfo) -> tarfile.TarInfo:
        # Keep copied sources writable inside the container so pip metadata/build
        # steps can create temporary egg-info artifacts when needed.
        if info.isdir():
            info.mode = 0o777
        elif info.issym():
            pass
        elif info.mode & 0o111:
            info.mode = 0o777
        else:
            info.mode = 0o666
        return info

    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel in include_paths:
            src = project_root / rel
            if not src.exists():
                raise FileNotFoundError(f"missing required project file for retrieval install: {src}")
            tar.add(str(src), arcname=f"{target_name}/{rel.as_posix()}", filter=_writable_permissions)
    buf.seek(0)

    opam_docker.exec_cmd(["bash", "-lc", f"mkdir -p {shlex.quote(target_parent)}"])
    _put_archive(opam_docker, target_parent, buf.getvalue())

    pip_path = f"/home/{config.user}/miniconda/envs/rocq-ml/bin/pip"
    python_path = f"/home/{config.user}/miniconda/envs/rocq-ml/bin/python"
    server_bin_path = f"/home/{config.user}/miniconda/envs/rocq-ml/bin/rocq-ml-retrieval-server"

    _exec_in_container(
        opam_docker,
        ["bash", "-lc", f"chmod -R a+rwX {shlex.quote(target_dir)}"],
        user="root",
    )

    install_pkg_cmd = (
        "set -e\n"
        f"if [ -x {shlex.quote(pip_path)} ]; then\n"
        f"  {shlex.quote(pip_path)} install --no-build-isolation --no-deps {shlex.quote(target_dir)}\n"
        "else\n"
        f"  python3 -m pip install --no-build-isolation --no-deps {shlex.quote(target_dir)}\n"
        "fi\n"
    )
    _exec_in_container(opam_docker, ["bash", "-lc", install_pkg_cmd], user="root")

    if install_runtime_dependencies:
        dep_args = " ".join(shlex.quote(dep) for dep in RETRIEVAL_RUNTIME_DEPENDENCIES)
        install_deps_cmd = (
            "set -e\n"
            f"if [ -x {shlex.quote(pip_path)} ]; then\n"
            f"  {shlex.quote(pip_path)} install --no-cache-dir {dep_args}\n"
            "else\n"
            f"  python3 -m pip install --no-cache-dir {dep_args}\n"
            "fi\n"
        )
        _exec_in_container(opam_docker, ["bash", "-lc", install_deps_cmd], user="root")

        verify_deps_cmd = (
            "set -e\n"
            f"if [ -x {shlex.quote(python_path)} ]; then\n"
            f"  {shlex.quote(python_path)} - <<'PY'\n"
            "import faiss  # noqa: F401\n"
            "import fastapi  # noqa: F401\n"
            "import torch  # noqa: F401\n"
            "import transformers  # noqa: F401\n"
            "import uvicorn  # noqa: F401\n"
            "print('retrieval runtime deps: ok')\n"
            "PY\n"
            "else\n"
            "  python3 - <<'PY'\n"
            "import faiss  # noqa: F401\n"
            "import fastapi  # noqa: F401\n"
            "import torch  # noqa: F401\n"
            "import transformers  # noqa: F401\n"
            "import uvicorn  # noqa: F401\n"
            "print('retrieval runtime deps: ok')\n"
            "PY\n"
            "fi\n"
        )
        _exec_in_container(opam_docker, ["bash", "-lc", verify_deps_cmd], user="root")

    _exec_in_container(
        opam_docker,
        [
            "bash",
            "-lc",
            (
                "set -e\n"
                f"test -x {shlex.quote(server_bin_path)}\n"
                f"{shlex.quote(server_bin_path)} --help >/dev/null\n"
            ),
        ],
        user="root",
    )
    return target_dir


def _copy_env_embeddings_to_container(
    opam_docker: OpamDocker,
    *,
    env: str,
    local_embeddings_root: Path,
    container_embeddings_root: str,
) -> int:
    source_env_dir = local_embeddings_root / env
    pt_files = sorted(p for p in source_env_dir.glob("*.pt") if p.is_file())
    if not pt_files:
        raise FileNotFoundError(f"no embedding .pt files found for env={env} in {source_env_dir}")

    opam_docker.exec_cmd(["bash", "-lc", f"mkdir -p {shlex.quote(container_embeddings_root)}"])

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path in pt_files:
            tar.add(str(path), arcname=f"{env}/{path.name}")
    buf.seek(0)
    _put_archive(opam_docker, container_embeddings_root, buf.getvalue())
    return len(pt_files)


def _ensure_migrated_embeddings(
    *,
    envs: Sequence[str],
    input_root: Path,
    output_root: Path,
    migrate_script: Path,
    hf_dataset: str,
    hf_revision: str,
    hf_token: Optional[str],
    force_migrate: bool,
) -> None:
    missing: List[str] = []
    for env in envs:
        out_dir = output_root / env
        has_pt = out_dir.exists() and any(out_dir.glob("*.pt"))
        if force_migrate or not has_pt:
            missing.append(env)

    if not missing:
        return

    if not input_root.exists():
        raise FileNotFoundError(f"embedding input root not found: {input_root}")
    if not migrate_script.exists():
        raise FileNotFoundError(f"migrate script not found: {migrate_script}")

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] migrating embeddings for {len(missing)} env(s) to {output_root}")
    for env in missing:
        cmd = [
            "python",
            str(migrate_script),
            "--env",
            env,
            "--input-dir",
            str(input_root),
            "--output-dir",
            str(output_root),
            "--overwrite",
            "--hf-dataset",
            hf_dataset,
            "--hf-revision",
            hf_revision,
            "--streaming",
        ]
        if hf_token:
            cmd.extend(["--hf-token", hf_token])
        print(f"[migrate] env={env}")
        subprocess.run(cmd, check=True)


def _canonical_to_container_source_path(source_path_canonical: str, opam_env_path: str) -> str:
    parts = PurePosixPath(source_path_canonical).parts
    if not parts:
        raise ValueError(f"invalid canonical source path: {source_path_canonical!r}")

    if parts[0] == "Corelib":
        rel = PurePosixPath(*parts[1:]) if len(parts) > 1 else PurePosixPath("")
        if not str(rel):
            raise ValueError(f"invalid Corelib canonical path: {source_path_canonical!r}")
        return str(PurePosixPath(opam_env_path) / "lib" / "coq" / "theories" / rel)

    return str(PurePosixPath(opam_env_path) / "lib" / "coq" / "user-contrib" / PurePosixPath(source_path_canonical))


def _load_sources(sources_path: Path) -> List[SourceInfo]:
    table = pq.read_table(sources_path, columns=["source_id", "source_idx", "source_path_canonical"])
    rows = table.to_pylist()

    output: List[SourceInfo] = []
    for fallback_idx, row in enumerate(rows):
        source_id = _to_str(row.get("source_id"))
        canonical = _to_str(row.get("source_path_canonical"))
        if not source_id or not canonical:
            raise RuntimeError(f"invalid sources row in {sources_path}: source_id/source_path_canonical required")
        source_idx = _to_int(row.get("source_idx"))
        if source_idx is None:
            source_idx = fallback_idx
        output.append(
            SourceInfo(
                source_id=source_id,
                source_idx=source_idx,
                source_path_canonical=canonical,
            )
        )

    output.sort(key=lambda x: x.source_idx)

    seen_ids: set[str] = set()
    for info in output:
        if info.source_id in seen_ids:
            raise RuntimeError(f"duplicate source_id in {sources_path}: {info.source_id}")
        seen_ids.add(info.source_id)

    return output


def _iter_toc_rows_with_source_id(
    toc_nodes_path: Path,
    *,
    required_columns: Sequence[str],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    pf = pq.ParquetFile(toc_nodes_path)
    for batch in pf.iter_batches(columns=list(required_columns), batch_size=100_000):
        for row in batch.to_pylist():
            source_id = _to_str(row.get("source_id"))
            if not source_id:
                raise RuntimeError(f"toc row missing source_id in {toc_nodes_path}")
            payload = dict(row)
            payload.pop("source_id", None)
            payload.pop("source_path_canonical", None)
            payload.pop("env_id", None)
            yield source_id, payload


def _split_toc_rows_without_source_id(toc_nodes_path: Path) -> List[List[Dict[str, Any]]]:
    pf = pq.ParquetFile(toc_nodes_path)
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    columns = [
        "preorder_index",
        "sibling_index",
        "depth",
        "kind",
        "name",
        "span_bp",
        "span_ep",
        "docstring",
        "data_uid",
        "data_extra_json",
    ]

    for batch in pf.iter_batches(columns=columns, batch_size=100_000):
        for row in batch.to_pylist():
            preorder_idx = _to_int(row.get("preorder_index"))
            if current and preorder_idx == 0:
                groups.append(current)
                current = []
            current.append(dict(row))

    if current:
        groups.append(current)

    return groups


def _node_from_flat_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    node: Dict[str, Any] = {}

    kind = _to_str(row.get("kind"))
    if kind is not None:
        node["kind"] = kind

    name = _to_str(row.get("name"))
    if name is not None:
        node["name"] = name

    span_bp = _to_int(row.get("span_bp"))
    span_ep = _to_int(row.get("span_ep"))
    if span_bp is not None or span_ep is not None:
        node["span"] = {"bp": span_bp, "ep": span_ep}

    data: Dict[str, Any] = {}
    raw_extra = row.get("data_extra_json")
    if isinstance(raw_extra, str) and raw_extra.strip():
        try:
            parsed_extra = json.loads(raw_extra)
        except json.JSONDecodeError:
            parsed_extra = {"raw": raw_extra}

        if isinstance(parsed_extra, dict):
            data.update(parsed_extra)
        else:
            data["value"] = parsed_extra

    data_uid = _to_str(row.get("data_uid"))
    if data_uid:
        data["uid"] = data_uid

    if data:
        node["data"] = data

    docstring = _to_str(row.get("docstring"))
    if isinstance(docstring, str) and docstring.strip():
        node["docstring"] = docstring

    node["members"] = []
    return node


def _prune_empty_members(node: Dict[str, Any]) -> None:
    members = node.get("members")
    if not isinstance(members, list):
        return
    for child in members:
        if isinstance(child, dict):
            _prune_empty_members(child)
    if not members:
        node.pop("members", None)


def _build_nested_toc(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda r: (
            _to_int(r.get("preorder_index")) or 0,
            _to_int(r.get("sibling_index")) or 0,
        ),
    )

    roots: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []

    for row in ordered:
        depth = _to_int(row.get("depth"))
        if depth is None or depth < 0:
            depth = 0
        if depth > len(stack):
            depth = len(stack)

        stack = stack[:depth]
        node = _node_from_flat_row(row)

        if depth == 0 or not stack:
            roots.append(node)
        else:
            stack[-1].setdefault("members", []).append(node)

        stack.append(node)

    for root in roots:
        _prune_empty_members(root)

    return roots


def _build_source_toc_payloads(
    *,
    toc_nodes_path: Path,
    sources: Sequence[SourceInfo],
    require_source_id: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    pf = pq.ParquetFile(toc_nodes_path)
    schema_names = set(pf.schema.names)

    source_ids = [s.source_id for s in sources]
    source_id_set = set(source_ids)

    base_columns = [
        "preorder_index",
        "sibling_index",
        "depth",
        "kind",
        "name",
        "span_bp",
        "span_ep",
        "docstring",
        "data_uid",
        "data_extra_json",
    ]

    if "source_id" in schema_names:
        required_columns = ["source_id", "source_path_canonical", "env_id", *base_columns]
        # Keep only columns that exist (older parquet may miss env_id/source_path_canonical).
        required_columns = [c for c in required_columns if c in schema_names]

        out: Dict[str, List[Dict[str, Any]]] = {}

        current_source_id: Optional[str] = None
        current_rows: List[Dict[str, Any]] = []
        seen_completed: set[str] = set()

        for source_id, row in _iter_toc_rows_with_source_id(toc_nodes_path, required_columns=required_columns):
            if source_id not in source_id_set:
                # Ignore rows not referenced by sources.parquet for this env.
                continue

            if current_source_id is None:
                current_source_id = source_id

            if source_id != current_source_id:
                out[current_source_id] = _build_nested_toc(current_rows)
                seen_completed.add(current_source_id)

                if source_id in seen_completed:
                    raise RuntimeError(
                        "toc_nodes rows for the same source_id are not contiguous; "
                        "cannot stream safely"
                    )

                current_source_id = source_id
                current_rows = []

            current_rows.append(row)

        if current_source_id is not None:
            out[current_source_id] = _build_nested_toc(current_rows)

        for sid in source_ids:
            out.setdefault(sid, [])

        return out

    if require_source_id:
        raise RuntimeError(
            "toc_nodes.parquet has no `source_id` column. "
            "Re-export or patch HF toc_nodes with source provenance before running this script, "
            "or rerun with --allow-missing-source-id for best-effort sequential assignment."
        )

    print(
        "[warn] toc_nodes has no source_id column: using best-effort sequential assignment by preorder resets"
    )
    groups = _split_toc_rows_without_source_id(toc_nodes_path)

    if len(groups) > len(sources):
        raise RuntimeError(
            f"more toc groups ({len(groups)}) than sources ({len(sources)}); cannot assign sequentially"
        )

    if len(groups) != len(sources):
        print(
            f"[warn] toc groups ({len(groups)}) != sources ({len(sources)}). "
            "Assuming missing TOCs are trailing sources."
        )

    out = {s.source_id: [] for s in sources}
    for source, rows in zip(sources, groups):
        out[source.source_id] = _build_nested_toc(rows)

    return out


def _build_env_toc_payload(env: str, env_toc_path: Path) -> Dict[str, Any]:
    table = pq.read_table(env_toc_path)
    rows = table.to_pylist()

    ordered = sorted(
        rows,
        key=lambda r: (
            _to_int(r.get("preorder_index")) or 0,
            _to_int(r.get("sibling_index")) or 0,
        ),
    )

    root_id: Optional[str] = None
    nodes: List[Dict[str, Any]] = []
    file_index: Dict[str, str] = {}

    for row in ordered:
        node_id = _to_str(row.get("node_id"))
        if not node_id:
            continue

        parent_id = _to_str(row.get("parent_id"))
        if root_id is None and parent_id is None:
            root_id = node_id

        node: Dict[str, Any] = {
            "id": node_id,
            "type": _to_str(row.get("node_type")),
            "name": _to_str(row.get("name")),
            "path": _to_str(row.get("path")),
            "parent_id": parent_id,
            "children_ids": [
                child
                for child in (row.get("children_ids") if isinstance(row.get("children_ids"), list) else [])
                if isinstance(child, str)
            ],
            "one_liner": _to_str(row.get("one_liner")) or "",
        }

        entry_key = _to_str(row.get("entry_key"))
        if entry_key:
            node["entry_key"] = entry_key

        annotation_count = _to_int(row.get("annotation_count"))
        if annotation_count is not None:
            node["annotation_count"] = annotation_count

        nodes.append(node)

        node_type = _to_str(row.get("node_type"))
        if node_type == "file" and entry_key:
            file_index.setdefault(entry_key, node_id)

    if root_id is None:
        root_id = "dir:ROOT"

    return {
        "env": env,
        "generated_at": _utc_now_iso(),
        "root_id": root_id,
        "nodes": nodes,
        "file_index": file_index,
    }


def _local_image_exists(client: docker.DockerClient, repository: str, tag: str) -> bool:
    reference = f"{repository}:{tag}"
    return bool(client.images.list(filters={"reference": reference}))


def _discover_jobs(
    *,
    config_dir: Path,
    docker_client: docker.DockerClient,
    selected_envs: Optional[set[str]],
) -> Tuple[List[EnvJob], List[str]]:
    jobs: List[EnvJob] = []
    skipped: List[str] = []

    for config_path in sorted(config_dir.glob("*.yaml")):
        env = config_path.stem
        if selected_envs is not None and env not in selected_envs:
            continue

        config = OpamConfig.from_yaml(config_path)

        if not _local_image_exists(docker_client, config.name, config.tag):
            skipped.append(f"{env} (missing local image {config.name}:{config.tag})")
            continue

        jobs.append(EnvJob(env=env, config_path=config_path, config=config))

    return jobs, skipped


def _download_env_file(
    *,
    repo_id: str,
    revision: str,
    token: Optional[str],
    filename: str,
    cache_dir: Path,
) -> Path:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is not installed")

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
        filename=filename,
        local_dir=str(cache_dir),
    )
    return Path(path)


def _resolve_env_tables(
    *,
    env: str,
    hf_local_root: Optional[Path],
    hf_repo_id: str,
    hf_revision: str,
    hf_token: Optional[str],
    hf_cache_dir: Path,
) -> Tuple[Path, Path, Path]:
    relative = [
        f"{env}/sources.parquet",
        f"{env}/toc_nodes.parquet",
        f"{env}/env_toc.parquet",
    ]

    if hf_local_root is not None:
        env_dir = hf_local_root / env
        sources = env_dir / "sources.parquet"
        toc_nodes = env_dir / "toc_nodes.parquet"
        env_toc = env_dir / "env_toc.parquet"
        if sources.exists() and toc_nodes.exists() and env_toc.exists():
            return sources, toc_nodes, env_toc

    sources = _download_env_file(
        repo_id=hf_repo_id,
        revision=hf_revision,
        token=hf_token,
        filename=relative[0],
        cache_dir=hf_cache_dir,
    )
    toc_nodes = _download_env_file(
        repo_id=hf_repo_id,
        revision=hf_revision,
        token=hf_token,
        filename=relative[1],
        cache_dir=hf_cache_dir,
    )
    env_toc = _download_env_file(
        repo_id=hf_repo_id,
        revision=hf_revision,
        token=hf_token,
        filename=relative[2],
        cache_dir=hf_cache_dir,
    )
    return sources, toc_nodes, env_toc


def _resolve_output_image_ref(config: OpamConfig, repo_prefix: Optional[str], tag_suffix: str) -> Tuple[str, str]:
    if repo_prefix:
        base = config.name.split("/")[-1]
        repo = f"{repo_prefix.rstrip('/')}/{base}"
    else:
        repo = config.name

    tag = f"{config.tag}{tag_suffix}" if tag_suffix else config.tag
    return repo, tag


def _push_image(client: docker.DockerClient, repository: str, tag: str) -> Optional[str]:
    digest: Optional[str] = None
    for event in client.images.push(repository, tag=tag, stream=True, decode=True):
        if not isinstance(event, dict):
            continue
        err = event.get("error")
        if isinstance(err, str) and err:
            raise RuntimeError(f"docker push failed for {repository}:{tag}: {err}")
        aux = event.get("aux")
        if isinstance(aux, dict):
            maybe_digest = _to_str(aux.get("Digest"))
            if maybe_digest:
                digest = maybe_digest
    return digest


def _run_job(
    *,
    job: EnvJob,
    sources_path: Path,
    toc_nodes_path: Path,
    env_toc_path: Path,
    skip_prepare_targets: bool,
    update_rocq_ml: bool,
    output_repo_prefix: Optional[str],
    output_tag_suffix: str,
    push: bool,
    require_source_id: bool,
    install_retrieval_server: bool,
    install_retrieval_runtime_deps: bool,
    project_root: Path,
    container_retrieval_project_dir: Optional[str],
    copy_embeddings: bool,
    migrated_embeddings_root: Optional[Path],
    container_embeddings_root: str,
) -> Dict[str, Any]:
    config = job.config
    env = job.env

    sources = _load_sources(sources_path)
    source_tocs_by_id = _build_source_toc_payloads(
        toc_nodes_path=toc_nodes_path,
        sources=sources,
        require_source_id=require_source_id,
    )
    env_toc_payload = _build_env_toc_payload(env, env_toc_path)

    source_toc_payloads: Dict[str, str] = {}
    empty_source_toc_count = 0

    for source in sources:
        toc_payload_obj = source_tocs_by_id.get(source.source_id, [])
        if not toc_payload_obj:
            empty_source_toc_count += 1

        container_source_path = _canonical_to_container_source_path(
            source.source_path_canonical,
            config.opam_env_path,
        )
        toc_target_path = f"{container_source_path}.toc.json"
        toc_payload = json.dumps(toc_payload_obj, ensure_ascii=False)

        previous = source_toc_payloads.get(toc_target_path)
        if previous is not None and previous != toc_payload:
            raise RuntimeError(f"conflicting payload for source toc path {toc_target_path}")
        source_toc_payloads[toc_target_path] = toc_payload

    env_toc_target = str(PurePosixPath(config.opam_env_path) / "lib" / "coq" / f"{env}.toc.json")
    env_toc_payload_text = json.dumps(env_toc_payload, ensure_ascii=False)

    output_repo, output_tag = _resolve_output_image_ref(config, output_repo_prefix, output_tag_suffix)

    print(
        f"[env={env}] sources={len(sources)} source_toc_files={len(source_toc_payloads)} "
        f"empty_source_toc={empty_source_toc_count} output={output_repo}:{output_tag}"
    )

    opam_docker: Optional[OpamDocker] = None
    retrieval_project_dir: Optional[str] = None
    copied_embedding_files = 0
    try:
        opam_docker = OpamDocker(config, kill_clone=False, update_rocq_ml=update_rocq_ml)

        if install_retrieval_server:
            print(f"[env={env}] installing retrieval server package in container")
            retrieval_project_dir = _sync_retrieval_project_and_install(
                opam_docker,
                config,
                project_root=project_root,
                container_project_dir=container_retrieval_project_dir,
                install_runtime_dependencies=install_retrieval_runtime_deps,
            )

        if copy_embeddings:
            if migrated_embeddings_root is None:
                raise RuntimeError("copy_embeddings=True requires migrated_embeddings_root")
            print(f"[env={env}] copying migrated embeddings into container")
            copied_embedding_files = _copy_env_embeddings_to_container(
                opam_docker,
                env=env,
                local_embeddings_root=migrated_embeddings_root,
                container_embeddings_root=container_embeddings_root,
            )

        if not skip_prepare_targets:
            print(f"[env={env}] applying _prepare_targets")
            _prepare_targets(opam_docker, config)

        print(f"[env={env}] writing env TOC -> {env_toc_target}")
        opam_docker.write_file(env_toc_target, env_toc_payload_text, create_dir=True)

        print(f"[env={env}] writing {len(source_toc_payloads)} source TOC files")
        for idx, (target_path, payload) in enumerate(sorted(source_toc_payloads.items()), start=1):
            opam_docker.write_file(target_path, payload, create_dir=True)
            if idx % 500 == 0:
                print(f"[env={env}] wrote {idx}/{len(source_toc_payloads)} source TOC files")

        image = opam_docker.container.commit(output_repo, output_tag)
        image_id = image.id
        print(f"[env={env}] committed image {output_repo}:{output_tag} id={image_id}")
    finally:
        if opam_docker is not None:
            opam_docker.close()

    pushed_digest: Optional[str] = None
    if push:
        print(f"[env={env}] pushing {output_repo}:{output_tag}")
        pushed_digest = _push_image(docker_client, output_repo, output_tag)
        if pushed_digest:
            print(f"[env={env}] push complete digest={pushed_digest}")
        else:
            print(f"[env={env}] push complete")

    return {
        "env": env,
        "image": f"{output_repo}:{output_tag}",
        "sources": len(sources),
        "source_toc_files": len(source_toc_payloads),
        "empty_source_toc": empty_source_toc_count,
        "retrieval_project_dir": retrieval_project_dir,
        "copied_embedding_files": copied_embedding_files,
        "pushed_digest": pushed_digest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch local Pile-of-Rocq images with env/source TOC+docstrings from HF parquet, "
            "commit images, and optionally push to Docker Hub."
        )
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("Pile-of-rocq/config"),
        help="Directory containing Pile-of-Rocq env config YAMLs.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Restrict to specific env(s). Repeatable.",
    )

    parser.add_argument(
        "--hf-repo-id",
        default="theostos/pile-of-rocq",
        help="HF dataset repo id containing <env>/sources|toc_nodes|env_toc parquet files.",
    )
    parser.add_argument("--hf-revision", default="main", help="HF revision.")
    parser.add_argument("--hf-token", default=None, help="HF token (optional for public read).")
    parser.add_argument(
        "--hf-local-root",
        type=Path,
        default=None,
        help="Optional local mirror root; if set and files exist, avoids HF downloads.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=Path(".hf_toc_docstrings_patch"),
        help="HF download/cache directory.",
    )

    parser.add_argument(
        "--output-repo-prefix",
        default=None,
        help=(
            "Override output repository namespace/prefix. Example: --output-repo-prefix myorg "
            "maps theostos/coq-actuary -> myorg/coq-actuary."
        ),
    )
    parser.add_argument(
        "--output-tag-suffix",
        default="",
        help="Optional suffix appended to each output tag.",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Local LLM4Docq project root to copy/install retrieval server from.",
    )
    parser.add_argument(
        "--install-retrieval-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy/install retrieval server package in each patched container.",
    )
    parser.add_argument(
        "--install-retrieval-runtime-deps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Install retrieval runtime dependencies in-container "
            "(torch/transformers/faiss-cpu + FastAPI stack)."
        ),
    )
    parser.add_argument(
        "--container-retrieval-project-dir",
        default=None,
        help="Optional container path for copied retrieval project (default: /home/<user>/LLM4Docq).",
    )
    parser.add_argument(
        "--copy-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy migrated docstring embeddings for each env into the container.",
    )
    parser.add_argument(
        "--embeddings-input-root",
        type=Path,
        default=Path("envs/docstring_embeddings"),
        help="Local root containing precomputed (legacy) embeddings per env.",
    )
    parser.add_argument(
        "--migrated-embeddings-root",
        type=Path,
        default=Path("envs_v2/docstring_embeddings"),
        help="Local root containing migrated format-v2 embeddings per env.",
    )
    parser.add_argument(
        "--migrate-script",
        type=Path,
        default=Path("src/retrieval/migrate_embeddings_format.py"),
        help="Path to migration script used to upgrade legacy embeddings to format v2.",
    )
    parser.add_argument(
        "--force-migrate-embeddings",
        action="store_true",
        help="Force re-running migration for selected envs even if migrated files already exist.",
    )
    parser.add_argument(
        "--container-embeddings-root",
        default=None,
        help="Container root dir for embeddings (default: /home/<user>/docstring_embeddings).",
    )

    parser.add_argument(
        "--skip-prepare-targets",
        action="store_true",
        help="Skip _prepare_targets (_CoqProject/copy operations) before writing TOC files.",
    )
    parser.add_argument(
        "--update-rocq-ml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OpamDocker(update_rocq_ml=...) before patching.",
    )
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push committed images to registry after commit.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other envs after an env failure.",
    )
    parser.add_argument(
        "--allow-missing-source-id",
        action="store_true",
        help=(
            "Allow best-effort sequential TOC/source assignment when toc_nodes has no source_id column. "
            "By default this is a hard error to prevent misassignment."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_dir = _resolve_path(args.config_dir, must_exist=True)
    project_root = _resolve_path(args.project_root, must_exist=True)
    hf_local_root = _resolve_path(args.hf_local_root, must_exist=True) if args.hf_local_root else None
    hf_cache_dir = _resolve_path(args.hf_cache_dir, must_exist=False)
    embeddings_input_root = (
        _resolve_path(args.embeddings_input_root, must_exist=False) if args.copy_embeddings else None
    )
    migrated_embeddings_root = (
        _resolve_path(args.migrated_embeddings_root, must_exist=False) if args.copy_embeddings else None
    )
    migrate_script = _resolve_path(args.migrate_script, must_exist=True) if args.copy_embeddings else None

    selected_envs = set(args.env) if args.env else None

    docker_client = docker.from_env()
    jobs, skipped = _discover_jobs(
        config_dir=config_dir,
        docker_client=docker_client,
        selected_envs=selected_envs,
    )

    if skipped:
        print(f"[info] skipped {len(skipped)} config(s) without local image")
        for line in skipped:
            print(f"  - {line}")

    if not jobs:
        raise SystemExit("No matching configs with local images found.")

    print(f"[info] processing {len(jobs)} env(s)")

    if args.copy_embeddings:
        assert embeddings_input_root is not None
        assert migrated_embeddings_root is not None
        assert migrate_script is not None
        _ensure_migrated_embeddings(
            envs=[job.env for job in jobs],
            input_root=embeddings_input_root,
            output_root=migrated_embeddings_root,
            migrate_script=migrate_script,
            hf_dataset=args.hf_repo_id,
            hf_revision=args.hf_revision,
            hf_token=args.hf_token,
            force_migrate=bool(args.force_migrate_embeddings),
        )

    successes: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    for job in jobs:
        try:
            sources_path, toc_nodes_path, env_toc_path = _resolve_env_tables(
                env=job.env,
                hf_local_root=hf_local_root,
                hf_repo_id=args.hf_repo_id,
                hf_revision=args.hf_revision,
                hf_token=args.hf_token,
                hf_cache_dir=hf_cache_dir,
            )

            result = _run_job(
                job=job,
                sources_path=sources_path,
                toc_nodes_path=toc_nodes_path,
                env_toc_path=env_toc_path,
                skip_prepare_targets=args.skip_prepare_targets,
                update_rocq_ml=bool(args.update_rocq_ml),
                output_repo_prefix=args.output_repo_prefix,
                output_tag_suffix=args.output_tag_suffix,
                push=bool(args.push),
                require_source_id=not bool(args.allow_missing_source_id),
                install_retrieval_server=bool(args.install_retrieval_server),
                install_retrieval_runtime_deps=bool(args.install_retrieval_runtime_deps),
                project_root=project_root,
                container_retrieval_project_dir=args.container_retrieval_project_dir,
                copy_embeddings=bool(args.copy_embeddings),
                migrated_embeddings_root=migrated_embeddings_root if args.copy_embeddings else None,
                container_embeddings_root=(
                    args.container_embeddings_root or f"/home/{job.config.user}/docstring_embeddings"
                ),
            )
            successes.append(result)
        except Exception as exc:  # pragma: no cover
            msg = f"{type(exc).__name__}: {exc}"
            failures.append((job.env, msg))
            print(f"[error] env={job.env} failed: {msg}")
            if not args.continue_on_error:
                break

    print("\n[summary]")
    print(f"successes={len(successes)} failures={len(failures)}")

    for item in successes:
        print(
            "  - "
            f"{item['env']} -> {item['image']} "
            f"source_toc_files={item['source_toc_files']} "
            f"empty_source_toc={item['empty_source_toc']} "
            f"embedding_files={item['copied_embedding_files']}"
        )

    for env, err in failures:
        print(f"  - FAIL {env}: {err}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
