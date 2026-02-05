import time
from typing import List, Dict, Any, Tuple
from pymongo import MongoClient
from pymongo.collection import Collection
import numpy as np


class MongoDBBackend:
    """
    MongoDB Atlas backend for vector search.
    Uses Atlas Vector Search for similarity search.
    """

    def __init__(
        self,
        connection_string: str,
        database_name: str = "vectordb",
        collection_name: str = "documents",
    ):
        """
        Initialize MongoDB Atlas connection.
        
        Args:
            connection_string: MongoDB Atlas connection string
            database_name: Name of the database
            collection_name: Name of the collection
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection: Collection = self.db[collection_name]
        self.collection_name = collection_name
        self.database_name = database_name
        self._index_ready = False

    def recreate_collection(self, dim: int):
        """
        Recreate collection to ensure clean state for benchmarks.
        """
        # Clear documents instead of dropping the collection so Atlas Search index remains.
        # Dropping the collection deletes the search index definition in Atlas.
        if self.collection_name in self.db.list_collection_names():
            self.collection.delete_many({})
        # Collection will be created automatically on first insert if missing.

    def upsert(self, rows: List[Dict[str, Any]], batch_size: int = 1000):
        """
        Insert vectors into MongoDB Atlas.
        
        Args:
            rows: List of dicts with {"id": int, "text": str, "embedding": np.array/list}
            batch_size: Number of documents to insert per batch
        """
        total = len(rows)
        for start in range(0, total, batch_size):
            chunk = rows[start:start + batch_size]
            
            documents = []
            for r in chunk:
                embedding = r["embedding"]
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                
                doc = {
                    "_id": int(r["id"]),
                    "text": r["text"],
                    "embedding": embedding,
                }
                
                # Add topic if present
                if "topic" in r:
                    doc["topic"] = r["topic"]
                
                documents.append(doc)
            
            # Use bulk_write for better performance
            if documents:
                self.collection.insert_many(documents, ordered=False)

    def search(self, query_vector, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Perform vector search using Atlas Vector Search.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (id, text, score) tuples
        """
        qvec = query_vector
        if hasattr(qvec, "tolist"):
            qvec = qvec.tolist()
        
        # Validate query vector
        if not qvec or len(qvec) == 0:
            raise ValueError("Query vector is empty")
        
        # Check if collection has documents
        doc_count = self.collection.count_documents({})
        if doc_count == 0:
            # No documents in collection - return empty results
            return []
        
        vector_search_pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": qvec,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        knn_beta_pipeline = [
            {
                "$search": {
                    "index": "vector_index",
                    "knnBeta": {
                        "path": "embedding",
                        "vector": qvec,
                        "k": top_k,
                    },
                },
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
        ]
        
        import warnings

        def run_pipeline(pipeline):
            found = []
            for doc in self.collection.aggregate(pipeline):
                _id = int(doc["_id"])
                text = doc.get("text", "")
                score = float(doc.get("score", 0.0))
                found.append((_id, text, score))
            return found

        error_messages = []
        results = []
        try:
            results = run_pipeline(vector_search_pipeline)
        except Exception as e:
            error_messages.append(f"$vectorSearch failed: {e}")

        if not results:
            try:
                results = run_pipeline(knn_beta_pipeline)
            except Exception as e:
                error_messages.append(f"$search knnBeta failed: {e}")

        if results:
            return results

        if error_messages:
            warnings.warn(
                "MongoDB vector search failed. "
                + " ".join(error_messages)
                + f" Collection has {doc_count} documents."
            )
            return []

        # If no results but documents exist and index should be ready, warn
        if doc_count > 0 and self._index_ready:
            warnings.warn(
                f"MongoDB vector search returned 0 results despite {doc_count} documents in collection. "
                f"Check that the 'vector_index' exists and is configured correctly in Atlas."
            )

        return []

    def wait_for_index(self, query_vector, top_k: int = 5, timeout_s: int = 60, interval_s: float = 2.0) -> bool:
        """
        Wait for Atlas Search index to be queryable after inserts.
        Returns True if index appears ready (returns results), else False.
        """
        import time

        start = time.time()
        while time.time() - start < timeout_s:
            results = self.search(query_vector, top_k=top_k)
            if results:
                self._index_ready = True
                return True
            time.sleep(interval_s)

        return False

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

