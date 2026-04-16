# LLM4Docq

Docstring/retrieval tooling built on top of the Pile-of-Rocq data pipeline.

## Relationship with Pile-of-Rocq

This repository extends:

- https://github.com/LLM4Rocq/Pile-of-rocq

Use the upstream README for extraction/export details and dataset exploration conventions:

- https://github.com/LLM4Rocq/Pile-of-rocq/blob/main/README.md

## Quick Start (Use Existing Data)

If you only want to consume docstrings/retrieval, you do not need to run the generation pipeline below.

### 1) Explore docstrings from Hugging Face

Public dataset:

- https://huggingface.co/datasets/theostos/pile-of-rocq

Structure (single-repo layout):

- one folder per environment: `<env>/`
- one parquet table per artifact:
  - `sources.parquet`
  - `toc_nodes.parquet` (includes `docstring`, `source_id`, spans, kind/name)
  - `env_toc.parquet`
  - `proofs.parquet`, `proof_steps.parquet`, `step_deps.parquet`, `proof_axioms.parquet`, `env_metadata.parquet`
- HF configs are `<env>-<table>`, for example `coq-mathcomp-toc_nodes`

Minimal Python snippet:

```python
from datasets import load_dataset

repo = "theostos/pile-of-rocq"
env = "coq-mathcomp"

toc_nodes = load_dataset(repo, f"{env}-toc_nodes", split="train")
print("rows:", len(toc_nodes))
print("columns:", toc_nodes.column_names)

target = next(
    row for row in toc_nodes
    if (row.get("docstring") or "").strip() and row.get("kind") in {"definition", "notation", "theorem"}
)
print(target["kind"], target.get("name"))
print((target.get("docstring") or "")[:300])
```

### 2) Start retrieval server from Docker image

Docker images are published at:

- https://hub.docker.com/u/theostos

Run one image and expose retrieval on port `8010`:

```bash
docker run --rm -p 8010:8010 theostos/coq-mathcomp:9.0-2.5.0 rocq-ml-retrieval-server \
  --embeddings-root /home/rocq/docstring_embeddings \
  --host 0.0.0.0 \
  --port 8010
```

Optional quick check:

```bash
curl -s http://127.0.0.1:8010/envs
```

## Generate / Update Data

Pipeline flow:
1. `script/chunk.py`
2. `script/compute_docstrings.py`
3. `script/merge_docstrings_exports.py`
4. `script/export_hf.py`
5. `script/peek_env_toc.py` (inspection/debug)

## Chunk

`chunk.py`:
- reads env JSON/JSONL export files
- extracts target TOC elements and removes proof text
- injects UIDs and builds annotation chunks
- writes the same target UIDs into TOC target nodes (`toc[*].data.uid`)
- writes per-env JSONL ready for `compute_docstrings.py`

## Directory assumptions

Typical setup used here:
- Base extraction dump: `../Pile-of-rocq/exported_v3/*.json` or `*.jsonl`
- Previous annotations (shards + merged): `exported_v3/`
- Merged annotations output: `exported_v3/merged/`

## Step 1: Build chunks + prefill cached docstrings

Single env test:

```bash
python -m script.chunk_ter \
  --input-dir ../Pile-of-rocq/exported_v3 \
  --output-dir chunk_ter_output \
  --env coq-actuary
```

All envs:

```bash
python -m script.chunk_ter \
  --input-dir ../Pile-of-rocq/exported_v3 \
  --output-dir chunk_ter_output \
  --workers 4
```

## Step 2: Compute missing docstrings + env-level TOC

```bash
python -m script.compute_docstrings \
  --input-dir chunk_ter_output \
  --output-dir exported_v3/annotated_jsonl_local \
  --state-dir .compute_docstrings_state_local \
  --config config/annotator/config.yaml \
  --templates config/annotator \
  --dir-template config/annotator/prompt_dir_one_liner.txt \
  --workers 4
```

Notes:
- This script is resumable (`--state-dir`).
- For multiple parallel jobs (HPC), use different `--output-dir` and `--state-dir` per job and split envs with repeated `--env`.
- Output rows keep `annotations` unchanged and also mirror them into `toc[*].docstring` using `toc[*].data.uid`.

## Step 3: Merge shards and check coverage

Merge all shard outputs:

```bash
python -m script.merge_docstrings_exports \
  --input-root exported_v3 \
  --output-dir exported_v3/merged
```

Merge + coverage audit against latest `Pile-of-rocq` dump:

```bash
python -m script.merge_docstrings_exports \
  --input-root exported_v3 \
  --output-dir exported_v3/merged \
  --baseline-export-dir ../Pile-of-rocq/export_v2 \
  --report-path exported_v3/merged/coverage_report.json
```

## Step 4: Export HF-ready parquet (docstring + env_toc)

```bash
python -m script.export_hf \
  --export-path ../Pile-of-rocq/export_v2 \
  --config-path ../Pile-of-rocq/config \
  --docstrings-path exported_v3/merged \
  --env-toc-path exported_v3/merged/toc \
  --output-path ../Pile-of-rocq/hf_export_docstrings_envtoc
```

Output per env contains normalized tables, including:
- `toc_nodes.parquet` with a single `docstring` field per TOC node
- `env_toc.parquet` with env-level tree nodes
- `export_hf.py` now prefers TOC-native `docstring`/`data.uid` when present and keeps fallback matching from merged annotation files.
- Most previously `*_json` columns are now native nested Arrow/Parquet fields (lists/structs).
- `diags` table is not exported.

## Step 5: Push to Hugging Face

Login:

```bash
hf auth login
```

Push all envs into one dataset repo:

```bash
python -m script.export_hf \
  --export-path export_v2 \
  --config-path Pile-of-rocq/config \
  --env-toc-path export_v2/toc \
  --push-to-hub \
  --hf-namespace <your-hf-namespace> \
  --push-layout single-repo \
  --repo-name pile-of-rocq
```

You do not need to pre-create the dataset repo manually; the script creates it if needed.
In `single-repo` mode, configs are `<env>-<table>` subsets (for example `coq-actuary-toc_nodes`).

## Step 6: Quick inspection

Browse env TOC:

```bash
python -m script.peek_env_toc \
  --export-root ../Pile-of-rocq/hf_export_docstrings_envtoc \
  --env coq-actuary \
  --node-path Corelib/Numbers
```

Inspect a file and sample docstrings:

```bash
python -m script.peek_env_toc \
  --export-root ../Pile-of-rocq/hf_export_docstrings_envtoc \
  --env coq-actuary \
  --file Corelib/Lists/List.v \
  --max-docstrings 20
```

## Retrieval

Retrieval tooling (embedding precompute + HTTP server + client) are in:

- `src/retrieval/`

See the standalone guide:

- [src/retrieval/README.md](src/retrieval/README.md)
