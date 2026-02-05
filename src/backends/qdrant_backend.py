import time
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
)


class QdrantBackend:
    """
    Qdrant backend for storing and searching embeddings.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6333,
        collection_name: str = "documents",
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def recreate_collection(self, dim: int):
        """
        Recreate collection to ensure clean state for benchmarks.
        """
        # delete if exists (ignore errors)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, rows: List[Dict[str, Any]], batch_size: int = 256):
        """
        rows: [{"id": int, "text": str, "embedding": np.array/list[float]}]

        Qdrant has a max request payload size, so we must insert in batches.
        batch_size=256 is a safe default for 384-dim vectors + text payload.
        """
        total = len(rows)
        for start in range(0, total, batch_size):
            chunk = rows[start:start + batch_size]

            points = []
            for r in chunk:
                points.append(
                    PointStruct(
                        id=int(r["id"]),
                        vector=r["embedding"].tolist() if hasattr(r["embedding"], "tolist") else r["embedding"],
                        payload={
                            "text": r["text"],
                            "topic": r.get("topic"),
                        },
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points)


    def search(self, query_vector, top_k: int = 5):
        """
        Returns list of (id, text, score) ordered best-first.
        With cosine distance, Qdrant returns similarity score (higher is better).
        """
        qvec = query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=qvec,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in response.points:
            text = point.payload.get("text") if point.payload else None
            results.append((int(point.id), text, float(point.score)))

        return results

    def benchmark_search(self, query_vector, top_k: int = 10, repeats: int = 100) -> Dict[str, float]:
        """
        Simple latency benchmark: run the same search many times and measure ms.
        Returns p50 and p95 approx + avg.
        """
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
