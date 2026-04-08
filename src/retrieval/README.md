# Retrieval

This folder contains standalone retrieval tooling:

- `precompute_embeddings.py`:
  Compute and export per-library docstring embeddings (`.pt`) from Hugging Face datasets.
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
- Supported configs:
  - raw rows (`library`, `source`, `toc`)
  - source blobs (`entry_json`)
  - table pair (`<env>-sources` + `<env>-toc_nodes`)

## 2) Run Retrieval Server

```bash
python src/retrieval/server.py \
  --embeddings-root envs \
  --host 0.0.0.0 \
  --port 8010
```

Endpoints:
- `GET /health`
- `GET /envs`
- `POST /search`

`POST /search` body:

```json
{
  "query": "group homomorphism between finite groups",
  "env": "coq-mathcomp",
  "k": 5
}
```

Response includes:
- `docstring`
- `logical_path`
- `localization`
- nearest-neighbor `score`

FAISS cache:
- default cache dir is `<embeddings-root>/.faiss_cache`
- one cached index per env fingerprint, reused across restarts

## 3) Minimal Client

```bash
python src/retrieval/client.py \
  --base-url http://127.0.0.1:8010 \
  --env coq-mathcomp \
  --query "group homomorphism between finite groups" \
  --k 5
```
