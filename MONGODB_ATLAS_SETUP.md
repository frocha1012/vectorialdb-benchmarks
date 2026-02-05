# Adding MongoDB Atlas to the Benchmark

This guide will help you add MongoDB Atlas (cloud) as a 5th database to compare against the local databases.

---

## Step 1: Set Up MongoDB Atlas

### 1.1 Create MongoDB Atlas Account
1. Go to [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Sign up for a free account (free tier available)
3. Create a new organization and project

### 1.2 Create a Free Cluster
1. Click "Build a Database"
2. Choose **FREE (M0)** tier
3. Select a cloud provider and region (choose closest to you for best performance)
4. Name your cluster (e.g., "vector-benchmark")
5. Click "Create"

### 1.3 Configure Database Access
1. Go to **Database Access** (left sidebar)
2. Click "Add New Database User"
3. Choose "Password" authentication
4. Create username and password (save these!)
5. Set privileges to "Atlas admin" or "Read and write to any database"
6. Click "Add User"

### 1.4 Configure Network Access
1. Go to **Network Access** (left sidebar)
2. Click "Add IP Address"
3. For testing, click "Allow Access from Anywhere" (0.0.0.0/0)
   - ⚠️ **Security Note**: For production, restrict to your IP
4. Click "Confirm"

### 1.5 Get Connection String
1. Go to **Database** (left sidebar)
2. Click "Connect" on your cluster
3. Choose "Drivers"
4. Select **Python** and version **3.12 or later**
5. Copy the connection string
   - It looks like: `mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`
6. Replace `<username>` and `<password>` with your credentials

### 1.6 Enable Vector Search
1. Go to **Database** → Your cluster → "Browse Collections"
2. Create a database (e.g., `vectordb`)
3. Create a collection (e.g., `documents`)
4. Go to **Atlas Search** (left sidebar)
5. Click "Create Search Index"
6. Choose "JSON Editor"
7. Use this configuration:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

8. Name it `vector_index`
9. Click "Next" → "Create Search Index"
10. Wait for index to build (may take a few minutes)

---

## Step 2: Install MongoDB Driver

```bash
pip install pymongo
```

---

## Step 3: Create MongoDB Atlas Backend

Create a new file: `src/backends/mongodb_backend.py`

```python
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

    def recreate_collection(self, dim: int):
        """
        Recreate collection to ensure clean state for benchmarks.
        """
        # Drop collection if exists
        if self.collection_name in self.db.list_collection_names():
            self.db.drop_collection(self.collection_name)
        
        # Collection will be created automatically on first insert
        # Vector search index should already exist in Atlas

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
        
        # Atlas Vector Search aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": qvec,
                    "numCandidates": top_k * 10,  # Search more candidates for better results
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
        
        results = []
        for doc in self.collection.aggregate(pipeline):
            _id = int(doc["_id"])
            text = doc.get("text", "")
            score = float(doc.get("score", 0.0))
            results.append((_id, text, score))
        
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
```

---

## Step 4: Add Benchmark Function

Edit `src/benchmark/run_benchmark.py`:

### 4.1 Add Import
```python
from src.backends.mongodb_backend import MongoDBBackend
```

### 4.2 Add Benchmark Function
Add this function after the other benchmark functions:

```python
def benchmark_mongodb(rows, dim, query_vectors, ground_truth, top_k=5, connection_string: str = None) -> Dict[str, Any]:
    """
    Benchmark MongoDB Atlas.
    
    Args:
        connection_string: MongoDB Atlas connection string
                          If None, reads from environment variable MONGODB_ATLAS_URI
    """
    import os
    
    if connection_string is None:
        connection_string = os.getenv("MONGODB_ATLAS_URI")
        if connection_string is None:
            raise ValueError(
                "MongoDB Atlas connection string required. "
                "Set MONGODB_ATLAS_URI environment variable or pass connection_string parameter."
            )
    
    mongo = MongoDBBackend(
        connection_string=connection_string,
        database_name="vectordb",
        collection_name="documents"
    )
    mongo.recreate_collection(dim=dim)

    mem_before = get_memory_usage_mb()

    t0 = time.perf_counter()
    mongo.upsert(rows)
    t1 = time.perf_counter()
    insert_ms = (t1 - t0) * 1000.0

    mem_after = get_memory_usage_mb()

    latencies = []
    search_results = []
    for qv in query_vectors:
        s0 = time.perf_counter()
        results = mongo.search(qv, top_k=top_k)
        s1 = time.perf_counter()
        latencies.append((s1 - s0) * 1000.0)
        search_results.append(results)

    recall_metrics = compute_recall_metrics(search_results, ground_truth, k=top_k)

    throughput = measure_throughput(
        lambda qv, top_k=top_k: mongo.search(qv, top_k=top_k),
        query_vectors,
        top_k=top_k,
        duration_seconds=3.0
    )

    return {
        "db": "mongodb_atlas",
        "insert_ms": insert_ms,
        "insert_count": len(rows),
        "memory_mb": mem_after - mem_before,
        **summarize_latencies(latencies),
        **recall_metrics,
        **throughput,
    }
```

### 4.3 Update Main Function
In the `main()` function, add MongoDB Atlas to the benchmark:

```python
# 4) Run benchmarks
print("Running benchmarks...")
results: List[Dict[str, Any]] = []
results.append(benchmark_pgvector(rows, embedder.dim, query_vectors, ground_truth, top_k=5))
print("  ✓ pgvector")
results.append(benchmark_qdrant(rows, embedder.dim, query_vectors, ground_truth, top_k=5))
print("  ✓ qdrant")
results.append(benchmark_faiss(rows, embedder.dim, query_vectors, ground_truth, top_k=5))
print("  ✓ faiss")
results.append(benchmark_milvus(rows, embedder.dim, query_vectors, ground_truth, top_k=5))
print("  ✓ milvus")

# Add MongoDB Atlas (optional - only if connection string is set)
try:
    results.append(benchmark_mongodb(rows, embedder.dim, query_vectors, ground_truth, top_k=5))
    print("  ✓ mongodb_atlas")
except Exception as e:
    print(f"  ✗ mongodb_atlas (skipped: {e})")
```

---

## Step 5: Update Run Suite

Edit `src/benchmark/run_suite.py`:

### 5.1 Add Import
```python
from src.benchmark.run_benchmark import (
    benchmark_pgvector,
    benchmark_qdrant,
    benchmark_faiss,
    benchmark_milvus,
    benchmark_mongodb,  # Add this
    compute_ground_truth,
)
```

### 5.2 Update Main Loop
In the main loop where benchmarks are run:

```python
r_mv = benchmark_milvus(rows, embedder.dim, query_vectors, ground_truth, top_k=args.topk)
r_mv["n_dataset"] = n
suite_results.append(r_mv)

# Add MongoDB Atlas (optional)
try:
    r_mongo = benchmark_mongodb(rows, embedder.dim, query_vectors, ground_truth, top_k=args.topk)
    r_mongo["n_dataset"] = n
    suite_results.append(r_mongo)
except Exception as e:
    print(f"  MongoDB Atlas skipped: {e}")
```

### 5.3 Update Color Palette
In the dashboard functions, add MongoDB Atlas color:

```python
db_colors = {
    "faiss": "#2ca02c",
    "milvus": "#1f77b4",
    "pgvector": "#ff7f0e",
    "qdrant": "#d62728",
    "mongodb_atlas": "#00ED64",  # MongoDB green
}
```

---

## Step 6: Set Environment Variable

### Windows (PowerShell)
```powershell
$env:MONGODB_ATLAS_URI="mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
```

### Windows (CMD)
```cmd
set MONGODB_ATLAS_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
```

### Linux/Mac
```bash
export MONGODB_ATLAS_URI="mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
```

**Or create a `.env` file** (recommended):
```
MONGODB_ATLAS_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
```

Then install python-dotenv and load it:
```bash
pip install python-dotenv
```

Add to your benchmark files:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Step 7: Test MongoDB Atlas

### 7.1 Test Connection
Create `notebooks/test_mongodb.py`:

```python
import os
from dotenv import load_dotenv
from src.backends.mongodb_backend import MongoDBBackend
from src.pipeline.dataset import load_sample_dataset
from src.pipeline.embed import EmbeddingGenerator

load_dotenv()

def main():
    connection_string = os.getenv("MONGODB_ATLAS_URI")
    if not connection_string:
        print("Error: MONGODB_ATLAS_URI not set")
        return
    
    data = load_sample_dataset()
    texts = [d["text"] for d in data]
    
    embedder = EmbeddingGenerator()
    vectors = embedder.embed(texts)
    
    rows = []
    for item, vec in zip(data, vectors):
        rows.append({"id": item["id"], "text": item["text"], "embedding": vec})
    
    mongo = MongoDBBackend(connection_string=connection_string)
    mongo.recreate_collection(dim=embedder.dim)
    mongo.upsert(rows)
    
    print("Inserted rows:", len(rows))
    
    query_text = "What are embeddings used for?"
    query_vec = embedder.embed([query_text])[0]
    
    results = mongo.search(query_vec, top_k=5)
    print("\nQuery:", query_text)
    print("Top results:")
    for _id, text, score in results:
        print(f"  id={_id:02d} score={score:.4f} text={text}")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python -m notebooks.test_mongodb
```

### 7.2 Run Full Benchmark
```bash
python -m src.benchmark.run_benchmark --n 1000
```

---

## Step 8: Troubleshooting

### Issue: "Index not found"
- **Solution**: Make sure you created the vector search index in Atlas
- Check Atlas Search → Search Indexes → Should see `vector_index`

### Issue: "Authentication failed"
- **Solution**: Check username/password in connection string
- Verify database user has correct permissions

### Issue: "Connection timeout"
- **Solution**: Check Network Access in Atlas (should allow your IP)
- Verify connection string is correct

### Issue: "Vector dimension mismatch"
- **Solution**: Make sure index is configured for 384 dimensions (or your embedding dimension)
- Recreate index with correct dimensions

---

## Step 9: Compare Results

After running benchmarks, MongoDB Atlas will appear in:
- CSV results with `db="mongodb_atlas"`
- All dashboard visualizations
- Comparison charts

**Expected differences:**
- **Higher latency** than local databases (network overhead)
- **Lower throughput** (network bandwidth limits)
- **Similar accuracy** (should still achieve Recall@K = 1.0)

---

## Quick Start Checklist

- [ ] Create MongoDB Atlas account
- [ ] Create free cluster
- [ ] Configure database user
- [ ] Allow network access
- [ ] Create vector search index
- [ ] Copy connection string
- [ ] Install `pymongo`: `pip install pymongo`
- [ ] Create `src/backends/mongodb_backend.py`
- [ ] Update `run_benchmark.py` with benchmark function
- [ ] Update `run_suite.py` to include MongoDB
- [ ] Set `MONGODB_ATLAS_URI` environment variable
- [ ] Test connection with `test_mongodb.py`
- [ ] Run full benchmark

---

## Notes

- **Free tier limits**: M0 clusters have limited resources, may be slower than local
- **Network latency**: Expect +10-50ms additional latency vs local
- **Cost**: Free tier is sufficient for benchmarking, but has usage limits
- **Security**: For production, restrict network access to specific IPs

---

## Next Steps

Once MongoDB Atlas is working, you'll have:
- **4 Local databases**: FAISS, pgvector, Qdrant, Milvus
- **1 Cloud database**: MongoDB Atlas

This allows you to compare:
- Local vs Cloud performance
- Network overhead impact
- Managed service benefits

