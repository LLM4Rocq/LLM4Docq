from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval.server import EnvRuntime, RetrievalService, create_app  # noqa: E402


class DummyIndex:
    def __init__(self, size: int, dim: int = 8) -> None:
        self.ntotal = size
        self.d = dim

    def search(self, qvec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        ids = -np.ones((1, k), dtype=np.int64)
        scores = np.full((1, k), -np.inf, dtype=np.float32)
        n = min(k, self.ntotal)
        if n > 0:
            ids[0, :n] = np.arange(n, dtype=np.int64)
            scores[0, :n] = np.linspace(1.0, 0.1, n, dtype=np.float32)
        return scores, ids


class DummyEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), self.dim), dtype=np.float32)


def _build_service() -> RetrievalService:
    service = RetrievalService(
        embeddings_root=Path("/tmp/embeddings"),
        cache_dir=Path("/tmp/embeddings/.faiss_cache"),
        default_query_model="dummy-model",
        device="cpu",
        max_length=64,
        trust_remote_code=False,
        torch_dtype="auto",
    )

    runtime = EnvRuntime(
        env="coq-mathcomp",
        index=DummyIndex(size=6, dim=8),
        model_name="dummy-model",
        docstrings=[
            "d0",
            "d1",
            "d2",
            "d3",
            "d4",
            "d5",
        ],
        docstrings_raw=["r0", "r1", "r2", "r3", "r4", "r5"],
        statements=["s0", "s1", "s2", "s3", "s4", "s5"],
        logical_paths=[None, None, None, None, None, None],
        localizations=[{} for _ in range(6)],
        kinds=["definition", "notation", "definition", "lemma", None, "lemma"],
        names=["n0", "n1", "n2", "n3", "n4", "n5"],
        relative_paths=["p0", "p1", "p2", "p3", "p4", "p5"],
        libraries=["mathcomp", "mathcomp", "Stdlib", "Stdlib", "mathcomp", "Coquelicot"],
        fingerprint="dummy-fingerprint",
    )
    service._runtime_by_env["coq-mathcomp"] = runtime  # noqa: SLF001
    service.embedder_pool.get = lambda _model_name: DummyEmbedder(dim=runtime.index.d)  # type: ignore[assignment]
    return service


def test_get_libraries_route() -> None:
    app = create_app(_build_service())
    client = TestClient(app)

    resp = client.get("/get_libraries", params={"env": "coq-mathcomp"})
    assert resp.status_code == 200
    assert resp.json() == {
        "env": "coq-mathcomp",
        "libraries": ["Coquelicot", "Stdlib", "mathcomp"],
    }


def test_search_without_filters() -> None:
    app = create_app(_build_service())
    client = TestClient(app)

    resp = client.post(
        "/search",
        json={"env": "coq-mathcomp", "query": "group", "k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["returned"] == 3
    assert body["candidate_count"] == 6
    assert [x["library"] for x in body["results"]] == ["mathcomp", "mathcomp", "Stdlib"]


def test_search_filter_by_library() -> None:
    app = create_app(_build_service())
    client = TestClient(app)

    resp = client.post(
        "/search",
        json={
            "env": "coq-mathcomp",
            "query": "ring",
            "k": 10,
            "libraries": ["Stdlib"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidate_count"] == 2
    assert body["returned"] == 2
    assert all(x["library"] == "Stdlib" for x in body["results"])


def test_search_filter_by_kind() -> None:
    app = create_app(_build_service())
    client = TestClient(app)

    resp = client.post(
        "/search",
        json={
            "env": "coq-mathcomp",
            "query": "algebra",
            "k": 10,
            "kinds": ["lemma"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidate_count"] == 2
    assert body["returned"] == 2
    assert all(x["kind"] == "lemma" for x in body["results"])


def test_search_filter_by_library_and_kind() -> None:
    app = create_app(_build_service())
    client = TestClient(app)

    resp = client.post(
        "/search",
        json={
            "env": "coq-mathcomp",
            "query": "algebra",
            "k": 10,
            "libraries": ["mathcomp"],
            "kinds": ["definition"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidate_count"] == 1
    assert body["returned"] == 1
    assert body["results"][0]["library"] == "mathcomp"
    assert body["results"][0]["kind"] == "definition"


def test_search_unknown_library_rejected() -> None:
    app = create_app(_build_service())
    client = TestClient(app)

    resp = client.post(
        "/search",
        json={
            "env": "coq-mathcomp",
            "query": "x",
            "k": 5,
            "libraries": ["unknown-lib"],
        },
    )
    assert resp.status_code == 400
    detail = resp.json().get("detail", "")
    assert isinstance(detail, str)
    assert "Unknown libraries" in detail
