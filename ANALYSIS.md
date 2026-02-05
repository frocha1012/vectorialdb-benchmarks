# Metrics Analysis (Latest Run)

This note explains each metric in the benchmark output and whether the values
look reasonable based on the latest CSV:

`results/summary_20260205_001740.csv`

That run uses:
- HDF5 vector mode (`train` vectors, `test` queries)
- 769,382 documents, 1024-dim embeddings
- 1000 query vectors for latency/recall
- MongoDB Atlas skipped due to SSL handshake error
- `memory_mb` is populated because `psutil` was installed

**Total wall‑clock runtime (user observed):** ~40 minutes for the full run.

## Metric meanings

- **db**: Backend name (pgvector, qdrant, faiss, milvus).
- **insert_ms**: Total time to insert all vectors into the backend (ms).
- **insert_count**: Number of vectors inserted.
- **memory_mb**: Memory delta for the process during insert (MB). Requires `psutil`.
- **avg_ms / p50_ms / p95_ms / p99_ms**: Average and percentile query latencies
  over the measured query set (ms).
- **n**: Number of latency samples (one per query vector).
- **recall_at_k**: Average Recall@K vs FAISS exact search (K=5).
- **recall_std**: Std dev of Recall@K across queries.
- **n_queries**: Number of queries used to compute recall.
- **qps**: Queries per second from the throughput loop.
- **total_queries / duration_s**: Total queries and elapsed time for throughput.
- **n_dataset**: Dataset size used for this run.

## Does the latest run look reasonable?

## Pretty tables (latest run)

### Insert speed
| db | insert_ms | insert_s | vectors_per_s |
|---|---|---|---|
| pgvector | 538237.78 | 538.24 | 1429.45 |
| qdrant | 462721.21 | 462.72 | 1662.73 |
| faiss | 51426.88 | 51.43 | 14960.70 |
| milvus | 138366.07 | 138.37 | 5560.48 |

### Query latency distribution
| db | avg_ms | p50_ms | p95_ms | p99_ms | n_queries |
|---|---|---|---|---|---|
| pgvector | 702.80 | 671.93 | 893.40 | 1120.80 | 1000 |
| qdrant | 15.48 | 15.34 | 26.58 | 30.33 | 1000 |
| faiss | 72.56 | 71.56 | 78.96 | 88.76 | 1000 |
| milvus | 13.03 | 11.27 | 22.69 | 44.35 | 1000 |

### Recall@K
| db | recall_at_k | recall_std | n_queries |
|---|---|---|---|
| pgvector | 0.8676 | 0.1551 | 1000 |
| qdrant | 0.9962 | 0.0273 | 1000 |
| faiss | 1.0000 | 0.0000 | 1000 |
| milvus | 0.9916 | 0.0448 | 1000 |

### Throughput (QPS)
| db | qps | total_queries | duration_s |
|---|---|---|---|
| pgvector | 1.50 | 5 | 3.34 |
| qdrant | 66.31 | 199 | 3.00 |
| faiss | 13.28 | 40 | 3.01 |
| milvus | 98.08 | 295 | 3.01 |

### Memory delta
| db | memory_mb |
|---|---|
| pgvector | 2295.07 |
| qdrant | 3538.33 |
| faiss | 1777.60 |
| milvus | 1283.10 |

## How long it took (approx)

Approximate total time per backend = insert time + latency loop + throughput loop:

| db | insert_s | latency_loop_s | throughput_s | approx_total_s |
|---|---|---|---|---|
| pgvector | 538.24 | 702.80 | 3.34 | 1244.37 |
| qdrant | 462.72 | 15.48 | 3.00 | 481.20 |
| faiss | 51.43 | 72.56 | 3.01 | 127.00 |
| milvus | 138.37 | 13.03 | 3.01 | 154.40 |

These totals are approximate and exclude Python overhead and plotting.

### Insert time (`insert_ms`)
The ordering makes sense:
- **FAISS** is the fastest inserter (~51s) because it is in‑process and simple.
- **Milvus** is slower (~138s) but still much faster than pgvector/qdrant.
- **pgvector/qdrant** are the slowest (~462–538s), likely due to DB/network
  overhead and 1024‑dim vectors.
This behavior is plausible for large 769k inserts.

### Query latency (p50/p95/p99)
The latencies are consistent with backend expectations:
- **Milvus** and **Qdrant** are fastest (p50 ~11–15ms).
- **FAISS** is higher (~72ms) because this run uses CPU-only exact search on
  1024‑dim vectors and still runs inside the same process.
- **pgvector** is much slower (~672ms p50), which is typical for SQL execution
  plus high‑dim vectors without heavy indexing.
This ordering is expected.

### Throughput (`qps`)
The throughput matches latency ordering:
- **Milvus** and **Qdrant** are highest (~98 and ~66 QPS).
- **FAISS** is moderate (~13 QPS).
- **pgvector** is lowest (~1.5 QPS).
This aligns with the latency results and is reasonable.

### Recall@K
- **FAISS** shows 1.0 recall (expected for exact search).
- **Qdrant** (~0.996) and **Milvus** (~0.992) are near‑perfect.
- **pgvector** shows ~0.868 recall, which can happen because:
  - The search uses L2 distance, while FAISS ground truth uses cosine similarity
    (inner product on normalized vectors).
  - Differences in normalization can reduce overlap.

If you want recall comparisons to be tighter, you can ensure both pgvector and
FAISS use the same distance and normalization strategy.

### Memory (`memory_mb`)
Memory deltas are now recorded:
- **Qdrant** ~3538 MB
- **pgvector** ~2295 MB
- **FAISS** ~1778 MB
- **Milvus** ~1283 MB

These are process‑level deltas and can vary by backend and host load.

## Summary

Overall, the metrics are consistent with expected backend behavior for a large,
high‑dim dataset. The main caveat is:
- `pgvector` recall is lower due to metric mismatch (cosine vs L2).

## Quick summary (latest run)

| Category | Winner | Runner-up | Notes |
|----------|--------|-----------|-------|
| **Query Speed (Latency)** | 🥇 **Milvus** | 🥈 Qdrant | Milvus has lowest p50 latency |
| **Query Throughput (QPS)** | 🥇 **Milvus** | 🥈 Qdrant | Milvus handles most queries/second |
| **Insertion Speed** | 🥇 **FAISS** | 🥈 Milvus | FAISS is fastest at ingesting data |
| **Memory Efficiency** | 🥇 **Milvus** | 🥈 FAISS | Milvus uses least memory |
| **Accuracy (Recall@K)** | 🥇 **FAISS** | 🥈 Qdrant | FAISS is exact; Qdrant is near-perfect |

