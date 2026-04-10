# LLM4Docq

Docstring pipeline for Rocq/Coq sources.

The flow is:
1. `script/chunk.py`
2. `script/compute_docstrings.py`
3. `script/merge_docstrings_exports.py`
4. `script/enrich_export_v2_docstrings.py` (optional, in-place backfill of `export_v2`)
5. `script/export_hf.py`
6. `script/peek_env_toc.py` (inspection/debug)

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

## Step 4 (optional): Backfill `export_v2` in place with annotations + TOC docstrings

Use this when you want `export_v2/*.jsonl` to directly include:
- `annotations` copied from a docstring export (for example `annotated_jsonl_4`)
- TOC target `data.uid`
- TOC target `docstring`

All envs, in place:

```bash
python script/enrich_export_v2_docstrings.py \
  --export-dir export_v2 \
  --annotated-dir annotated_jsonl_4 \
  --strict-missing
```

Single env:

```bash
python script/enrich_export_v2_docstrings.py \
  --export-dir export_v2 \
  --annotated-dir annotated_jsonl_4 \
  --env coq-actuary \
  --strict-missing
```

`--strict-missing` aborts if any `export_v2` row has no matching annotation row.

## Step 5: Export HF-ready parquet (docstring + env_toc)

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

## Step 6: Push to Hugging Face

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

## Step 7: Quick inspection

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
