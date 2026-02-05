import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from src.pipeline.dataset import (
    load_synthetic_dataset,
    load_file_dataset,
    load_hdf5_dataset,
    load_hdf5_vectors,
)
from src.pipeline.embed import EmbeddingGenerator
from src.backends.pgvector_backend import PgVectorBackend
from src.backends.qdrant_backend import QdrantBackend
from src.backends.faiss_backend import FaissBackend
from src.backends.milvus_backend import MilvusBackend
from src.backends.mongodb_backend import MongoDBBackend
from src.benchmark.metrics import (
    summarize_latencies,
    compute_recall_metrics,
    measure_throughput,
    get_memory_usage_mb,
)


def compute_ground_truth(rows, dim, query_vectors, top_k=5) -> List[List[Tuple[int, str, float]]]:
    """
    Compute ground truth results using FAISS exact search (IndexFlatIP).
    This serves as the baseline for Recall@K calculations.
    """
    fa = FaissBackend()
    fa.create_index(dim=dim)
    fa.upsert(rows)
    
    ground_truth = []
    for qv in query_vectors:
        results = fa.search(qv, top_k=top_k)
        ground_truth.append(results)
    
    return ground_truth


def benchmark_pgvector(rows, dim, query_vectors, ground_truth, top_k=5) -> Dict[str, Any]:
    pg = PgVectorBackend(host="127.0.0.1", port=5433, user="postgres", password="postgres", dbname="vectordb")
    pg.create_schema(dim=dim)
    pg.clear()

    # Measure memory before insert
    mem_before = get_memory_usage_mb()

    t0 = time.perf_counter()
    pg.upsert(rows)
    t1 = time.perf_counter()
    insert_ms = (t1 - t0) * 1000.0

    # Measure memory after insert
    mem_after = get_memory_usage_mb()

    # Collect search results and latencies
    latencies = []
    search_results = []
    for qv in query_vectors:
        s0 = time.perf_counter()
        results = pg.search(qv, top_k=top_k)
        s1 = time.perf_counter()
        latencies.append((s1 - s0) * 1000.0)
        search_results.append(results)

    # Compute Recall@K
    recall_metrics = compute_recall_metrics(search_results, ground_truth, k=top_k)

    # Measure throughput
    throughput = measure_throughput(
        lambda qv, top_k=top_k: pg.search(qv, top_k=top_k),
        query_vectors,
        top_k=top_k,
        duration_seconds=3.0
    )

    return {
        "db": "pgvector",
        "insert_ms": insert_ms,
        "insert_count": len(rows),
        "memory_mb": mem_after - mem_before,
        **summarize_latencies(latencies),
        **recall_metrics,
        **throughput,
    }


def benchmark_qdrant(rows, dim, query_vectors, ground_truth, top_k=5) -> Dict[str, Any]:
    qd = QdrantBackend(host="127.0.0.1", port=6333, collection_name="documents")
    qd.recreate_collection(dim=dim)

    mem_before = get_memory_usage_mb()

    t0 = time.perf_counter()
    qd.upsert(rows)
    t1 = time.perf_counter()
    insert_ms = (t1 - t0) * 1000.0

    mem_after = get_memory_usage_mb()

    latencies = []
    search_results = []
    for qv in query_vectors:
        s0 = time.perf_counter()
        results = qd.search(qv, top_k=top_k)
        s1 = time.perf_counter()
        latencies.append((s1 - s0) * 1000.0)
        search_results.append(results)

    recall_metrics = compute_recall_metrics(search_results, ground_truth, k=top_k)

    throughput = measure_throughput(
        lambda qv, top_k=top_k: qd.search(qv, top_k=top_k),
        query_vectors,
        top_k=top_k,
        duration_seconds=3.0
    )

    return {
        "db": "qdrant",
        "insert_ms": insert_ms,
        "insert_count": len(rows),
        "memory_mb": mem_after - mem_before,
        **summarize_latencies(latencies),
        **recall_metrics,
        **throughput,
    }

def benchmark_faiss(rows, dim, query_vectors, ground_truth, top_k=5):
    fa = FaissBackend()
    fa.create_index(dim=dim)

    mem_before = get_memory_usage_mb()

    t0 = time.perf_counter()
    fa.upsert(rows)
    t1 = time.perf_counter()
    insert_ms = (t1 - t0) * 1000.0

    mem_after = get_memory_usage_mb()

    latencies = []
    search_results = []
    for qv in query_vectors:
        s0 = time.perf_counter()
        results = fa.search(qv, top_k=top_k)
        s1 = time.perf_counter()
        latencies.append((s1 - s0) * 1000.0)
        search_results.append(results)

    # FAISS with IndexFlatIP should have perfect recall (it's exact search)
    recall_metrics = compute_recall_metrics(search_results, ground_truth, k=top_k)

    throughput = measure_throughput(
        lambda qv, top_k=top_k: fa.search(qv, top_k=top_k),
        query_vectors,
        top_k=top_k,
        duration_seconds=3.0
    )

    return {
        "db": "faiss",
        "insert_ms": insert_ms,
        "insert_count": len(rows),
        "memory_mb": mem_after - mem_before,
        **summarize_latencies(latencies),
        **recall_metrics,
        **throughput,
    }

def benchmark_milvus(rows, dim, query_vectors, ground_truth, top_k=5):
    mv = MilvusBackend(host="127.0.0.1", port="19530", collection_name="documents")
    mv.recreate_collection(dim=dim)

    mem_before = get_memory_usage_mb()

    t0 = time.perf_counter()
    mv.upsert(rows)
    t1 = time.perf_counter()
    insert_ms = (t1 - t0) * 1000.0

    mem_after = get_memory_usage_mb()

    latencies = []
    search_results = []
    for qv in query_vectors:
        s0 = time.perf_counter()
        results = mv.search(qv, top_k=top_k)
        s1 = time.perf_counter()
        latencies.append((s1 - s0) * 1000.0)
        search_results.append(results)

    recall_metrics = compute_recall_metrics(search_results, ground_truth, k=top_k)

    throughput = measure_throughput(
        lambda qv, top_k=top_k: mv.search(qv, top_k=top_k),
        query_vectors,
        top_k=top_k,
        duration_seconds=3.0
    )

    return {
        "db": "milvus",
        "insert_ms": insert_ms,
        "insert_count": len(rows),
        "memory_mb": mem_after - mem_before,
        **summarize_latencies(latencies),
        **recall_metrics,
        **throughput,
    }


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

    # Wait for Atlas Search index to be queryable after insert
    if query_vectors:
        mongo.wait_for_index(query_vectors[0], top_k=top_k, timeout_s=90, interval_s=3.0)

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


def main():
    # 1) Load dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Dataset size")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to a folder of text files")
    parser.add_argument("--dataset_hdf5", type=str, default=None, help="Path to an HDF5 dataset file")
    parser.add_argument(
        "--dataset_hdf5_text_key",
        type=str,
        default=None,
        help="Dataset name or field for text inside the HDF5 file",
    )
    parser.add_argument(
        "--dataset_hdf5_vectors_key",
        type=str,
        default=None,
        help="Dataset name for embeddings inside the HDF5 file",
    )
    parser.add_argument(
        "--dataset_hdf5_query_key",
        type=str,
        default=None,
        help="Dataset name for query embeddings inside the HDF5 file",
    )
    parser.add_argument(
        "--dataset_hdf5_max_queries",
        type=int,
        default=5,
        help="Max query vectors to use for HDF5 embeddings",
    )
    parser.add_argument("--dataset_max_docs", type=int, default=200, help="Max files to load")
    parser.add_argument("--dataset_min_chars", type=int, default=50, help="Min characters per doc")
    parser.add_argument("--dataset_max_chars", type=int, default=1000, help="Max characters per doc")
    parser.add_argument(
        "--dataset_exts",
        type=str,
        default="txt,md,markdown,rst,json,jsonl,csv",
        help="Comma-separated list of file extensions",
    )
    args = parser.parse_args()

    query_vectors = None
    embedder_dim = None

    if args.dataset_hdf5:
        if args.dataset_hdf5_vectors_key or args.dataset_hdf5_query_key:
            data, query_vectors, embedder_dim = load_hdf5_vectors(
                args.dataset_hdf5,
                vectors_key=args.dataset_hdf5_vectors_key,
                query_key=args.dataset_hdf5_query_key,
                max_docs=args.dataset_max_docs,
                max_queries=args.dataset_hdf5_max_queries,
            )
        else:
            data = load_hdf5_dataset(
                args.dataset_hdf5,
                text_key=args.dataset_hdf5_text_key,
                max_docs=args.dataset_max_docs,
                min_chars=args.dataset_min_chars,
                max_chars=args.dataset_max_chars,
            )
            # Fallback: if text load yields nothing, try vector datasets
            if not data:
                data, query_vectors, embedder_dim = load_hdf5_vectors(
                    args.dataset_hdf5,
                    vectors_key=args.dataset_hdf5_vectors_key,
                    query_key=args.dataset_hdf5_query_key,
                    max_docs=args.dataset_max_docs,
                    max_queries=args.dataset_hdf5_max_queries,
                )
    elif args.dataset_dir:
        exts = [e.strip() for e in args.dataset_exts.split(",") if e.strip()]
        data = load_file_dataset(
            args.dataset_dir,
            exts=exts,
            max_docs=args.dataset_max_docs,
            min_chars=args.dataset_min_chars,
            max_chars=args.dataset_max_chars,
        )
    else:
        data = load_synthetic_dataset(n=args.n)

    if not data:
        raise ValueError("Dataset is empty. Check dataset_dir or filters.")

    rows = []
    if data and "embedding" in data[0]:
        rows = data
        if embedder_dim is None:
            embedder_dim = len(rows[0]["embedding"])
    else:
        texts = [d["text"] for d in data]

        # 2) Generate embeddings once
        embedder = EmbeddingGenerator()
        vectors = embedder.embed(texts)
        embedder_dim = embedder.dim

        for item, vec in zip(data, vectors):
            rows.append({"id": item["id"], "text": item["text"], "embedding": vec})

    # 3) Build query set
    if query_vectors is None:
        if args.dataset_dir:
            query_texts = [t[:200] for t in texts[:5]]
            if len(query_texts) < 5:
                query_texts.append("What is this document about?")
        else:
            query_texts = [
                "What are embeddings used for?",
                "Explain vector databases",
                "How does Docker help reproducibility?",
                "What is similarity search?",
                "How does RAG help LLMs?",
            ]
        query_vectors = embedder.embed(query_texts)

    # 3.5) Compute ground truth using FAISS exact search
    print("Computing ground truth with FAISS exact search...")
    ground_truth = compute_ground_truth(rows, embedder_dim, query_vectors, top_k=5)

    # 4) Run benchmarks
    print("Running benchmarks...")
    results: List[Dict[str, Any]] = []
    results.append(benchmark_pgvector(rows, embedder_dim, query_vectors, ground_truth, top_k=5))
    print("  OK pgvector")
    results.append(benchmark_qdrant(rows, embedder_dim, query_vectors, ground_truth, top_k=5))
    print("  OK qdrant")
    results.append(benchmark_faiss(rows, embedder_dim, query_vectors, ground_truth, top_k=5))
    print("  OK faiss")
    results.append(benchmark_milvus(rows, embedder_dim, query_vectors, ground_truth, top_k=5))
    print("  OK milvus")
    
    # MongoDB Atlas (optional - only if connection string is set)
    try:
        results.append(benchmark_mongodb(rows, embedder_dim, query_vectors, ground_truth, top_k=5))
        print("  OK mongodb_atlas")
    except Exception as e:
        print(f"  x mongodb_atlas (skipped: {e})")

    # Single-run dataset size for charts
    dataset_size = len(rows)
    for r in results:
        r["n_dataset"] = dataset_size


    # 5) Save outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = f"results/run_{ts}.json"
    out_csv = f"results/summary_{ts}.csv"
    out_png = f"results/latency_{ts}.png"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "results": results}, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    # Copy summary CSV to site/public for index.html
    try:
        from pathlib import Path
        site_dir = Path(__file__).parent.parent.parent / "site" / "public"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "benchmark_results.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        print("CSV file copied to: ", site_dir / "benchmark_results.csv")
    except Exception as e:
        print("Failed to copy CSV to site/public:", e)

    # 6) Plot comparisons
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Latency comparison
    ax1 = axes[0, 0]
    x = df["db"].tolist()
    p50 = df["p50_ms"].tolist()
    p95 = df["p95_ms"].tolist()
    p99 = df["p99_ms"].tolist()
    ax1.plot(x, p50, marker="o", label="p50")
    ax1.plot(x, p95, marker="s", label="p95")
    ax1.plot(x, p99, marker="^", label="p99")
    ax1.set_xlabel("Database")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Query Latency (p50/p95/p99)")
    ax1.legend()
    ax1.grid(True)
    
    # Recall@K comparison
    ax2 = axes[0, 1]
    recall = df["recall_at_k"].tolist()
    ax2.bar(x, recall, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_xlabel("Database")
    ax2.set_ylabel("Recall@K")
    ax2.set_title("Search Accuracy (Recall@K)")
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, axis='y')
    
    # Throughput (QPS) comparison
    ax3 = axes[1, 0]
    qps = df["qps"].tolist()
    ax3.bar(x, qps, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax3.set_xlabel("Database")
    ax3.set_ylabel("Queries per Second")
    ax3.set_title("Query Throughput (QPS)")
    ax3.grid(True, axis='y')
    
    # Memory usage comparison
    ax4 = axes[1, 1]
    memory = df["memory_mb"].tolist()
    ax4.bar(x, memory, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax4.set_xlabel("Database")
    ax4.set_ylabel("Memory (MB)")
    ax4.set_title("Memory Usage")
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print("\nSaved:")
    print(" -", out_json)
    print(" -", out_csv)
    print(" -", out_png)
    print("\nSummary:")
    print(df)


if __name__ == "__main__":
    main()
