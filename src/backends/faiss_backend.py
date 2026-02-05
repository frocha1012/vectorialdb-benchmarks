import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss


class FaissBackend:
    """
    FAISS baseline backend (local, in-process).
    Uses cosine similarity via inner product on normalized vectors.

    - We normalize embeddings (already normalized by your embedder)
    - We use IndexFlatIP (exact search) for correctness baseline.
      Later you can switch to HNSW/IVF for ANN.
    """

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.id_map: Optional[faiss.IndexIDMap] = None
        self.text_by_id: Dict[int, str] = {}

    def create_index(self, dim: int):
        base = faiss.IndexFlatIP(dim)  # inner product (cosine if vectors are normalized)
        self.id_map = faiss.IndexIDMap(base)
        self.index = self.id_map
        self.text_by_id = {}

    def upsert(self, rows: List[Dict[str, Any]]):
        if self.index is None:
            raise RuntimeError("FAISS index not created. Call create_index(dim) first.")

        ids = np.array([int(r["id"]) for r in rows], dtype=np.int64)

        # Ensure float32 shape (n, dim)
        vecs = []
        for r in rows:
            v = r["embedding"]
            if hasattr(v, "tolist"):
                v = v.tolist()
            vecs.append(v)
            self.text_by_id[int(r["id"])] = r["text"]

        X = np.array(vecs, dtype=np.float32)

        # (Optional) Ensure normalized for cosine
        faiss.normalize_L2(X)

        self.index.add_with_ids(X, ids)

    def search(self, query_vector, top_k: int = 5) -> List[Tuple[int, str, float]]:
        if self.index is None:
            raise RuntimeError("FAISS index not created. Call create_index(dim) first.")

        q = query_vector
        if hasattr(q, "tolist"):
            q = q.tolist()

        Q = np.array([q], dtype=np.float32)
        faiss.normalize_L2(Q)

        scores, ids = self.index.search(Q, top_k)
        results = []
        for _id, score in zip(ids[0], scores[0]):
            if _id == -1:
                continue
            _id = int(_id)
            results.append((_id, self.text_by_id.get(_id), float(score)))
        return results

    def benchmark_search(self, query_vector, top_k: int = 10, repeats: int = 100) -> Dict[str, float]:
        times_ms = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = self.search(query_vector, top_k=top_k)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        times_ms.sort()
        p50 = times_ms[int(0.50 * (len(times_ms) - 1))]
        p95 = times_ms[int(0.95 * (len(times_ms) - 1))]
        avg = sum(times_ms) / len(times_ms)

        return {"avg_ms": avg, "p50_ms": p50, "p95_ms": p95, "repeats": repeats}
