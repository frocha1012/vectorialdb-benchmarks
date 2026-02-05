import time
from typing import List, Dict, Any, Tuple, Optional

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


class MilvusBackend:
    """
    Milvus backend (standalone in Docker).
    Stores vectors + metadata and performs similarity search.

    Notes:
    - Uses COSINE metric (works well with normalized embeddings).
    - Inserts in batches for stability.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: str = "19530",
        collection_name: str = "documents",
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None

    def connect(self):
        connections.connect(alias="default", host=self.host, port=self.port)

    def recreate_collection(self, dim: int):
        self.connect()

        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
        ]

        schema = CollectionSchema(fields, description="Vector benchmark documents")
        self.collection = Collection(name=self.collection_name, schema=schema)

        # Create index (HNSW is a common choice; stable for local tests)
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)

        # Load collection into memory
        self.collection.load()

    def upsert(self, rows: List[Dict[str, Any]], batch_size: int = 1000):
        if self.collection is None:
            raise RuntimeError("Collection not created. Call recreate_collection(dim) first.")

        total = len(rows)
        for start in range(0, total, batch_size):
            chunk = rows[start:start + batch_size]

            ids = [int(r["id"]) for r in chunk]
            embeddings = [
                (r["embedding"].tolist() if hasattr(r["embedding"], "tolist") else r["embedding"])
                for r in chunk
            ]
            topics = [(r.get("topic") or "")[:64] for r in chunk]
            texts = [(r.get("text") or "")[:1024] for r in chunk]

            self.collection.insert([ids, embeddings, topics, texts])

        self.collection.flush()

    def search(self, query_vector, top_k: int = 5) -> List[Tuple[int, str, float]]:
        if self.collection is None:
            raise RuntimeError("Collection not created. Call recreate_collection(dim) first.")

        qv = query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector

        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        results = self.collection.search(
            data=[qv],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"],
        )

        out = []
        for hit in results[0]:
            _id = int(hit.id)
            score = float(hit.score)  # for COSINE: higher is better
            text = hit.entity.get("text") if hit.entity else None
            out.append((_id, text, score))
        return out

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
