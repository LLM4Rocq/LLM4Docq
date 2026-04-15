# Retrieval

This folder contains standalone retrieval tooling:

- `precompute_embeddings.py`:
  Compute and export per-library docstring embeddings (`.pt`) from Hugging Face datasets.
- `migrate_embeddings_format.py`:
  Temporary migration utility to upgrade old `.pt` payloads to the new format without recomputing embeddings.
- `server.py`:
  FastAPI + FAISS HTTP retrieval server over those precomputed embeddings.
- `client.py`:
  Minimal Python HTTP client for `/search`.

## 1) Precompute Embeddings

Single env:

```bash
python src/retrieval/precompute_embeddings.py \
  --env coq-mathcomp \
  --hf-dataset theostos/pile-of-rocq \
  --model-name Qwen/Qwen3-Embedding-4B \
  --batch-size 16 \
  --device cuda:0 \
  --output-dir envs
```

All envs:

```bash
python src/retrieval/precompute_embeddings.py \
  --all-envs \
  --hf-dataset theostos/pile-of-rocq \
  --model-name Qwen/Qwen3-Embedding-4B \
  --batch-size 16 \
  --device cuda:0 \
  --output-dir envs
```

Notes:
- Input is Hugging Face dataset download only.
- Output payload format is `format_version=2`.
- Supported configs:
  - raw rows (`library`, `source`, `toc`)
  - source blobs (`entry_json`)
  - table pair (`<env>-sources` + `<env>-toc_nodes`)
- Each embedded `docstring` now includes the full extracted source statement + the original docstring.
- Localization is exported as:
  - `start_bp`, `end_bp`
  - `start_line`, `start_character`, `end_line`, `end_character`

## 1b) Migrate Old `.pt` Files (No Re-Embedding)

If you already computed embeddings with the previous format, migrate them once:

```bash
python src/retrieval/migrate_embeddings_format.py \
  --env coq-mathcomp \
  --input-dir envs \
  --output-dir envs_v2 \
  --hf-dataset theostos/pile-of-rocq
```

Then run the server with `--embeddings-root envs_v2`.
`server.py` now expects `format_version=2` payloads.

## 2) Run Retrieval Server

```bash
rocq-ml-retrieval-server \
  --embeddings-root envs \
  --host 0.0.0.0 \
  --port 8010
```

If you are running from source, install once to get CLI commands:

```bash
pip install -e .
```

Endpoints:
- `GET /health`
- `GET /envs`
- `GET /get_libraries?env=<env>`
- `POST /search`

`POST /search` body:

```json
{
  "query": "group homomorphism between finite groups",
  "env": "coq-mathcomp",
  "k": 5,
  "libraries": ["mathcomp"],
  "kinds": ["definition", "notation"]
}
```

- `libraries` is optional (default: all libraries in the env).
- `kinds` is optional (default: all kinds in the env).

Response includes:
- `docstring`
- `docstring_raw`
- `statement`
- `kind`
- `name`
- `logical_path`
- `localization`
- nearest-neighbor `score`

FAISS cache:
- default cache dir is `<embeddings-root>/.faiss_cache`
- one cached index per env fingerprint, reused across restarts

## 3) Minimal Client

```bash
rocq-ml-retrieval-client \
  --base-url http://127.0.0.1:8010 \
  --env coq-mathcomp \
  --query "group homomorphism between finite groups" \
  --k 5 \
  --library mathcomp \
  --kind definition
```
