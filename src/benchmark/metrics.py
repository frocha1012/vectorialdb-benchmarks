from typing import List, Dict, Set, Tuple
import math
import time
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def percentile(sorted_values: List[float], p: float) -> float:
    """
    p in [0,1]. sorted_values must be sorted ascending.
    """
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = p * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def summarize_latencies(times_ms: List[float]) -> Dict[str, float]:
    times_ms = sorted(times_ms)
    avg = sum(times_ms) / len(times_ms) if times_ms else float("nan")
    return {
        "avg_ms": avg,
        "p50_ms": percentile(times_ms, 0.50),
        "p95_ms": percentile(times_ms, 0.95),
        "p99_ms": percentile(times_ms, 0.99),
        "n": len(times_ms),
    }


def recall_at_k(
    predicted_ids: List[int],
    ground_truth_ids: List[int],
    k: int
) -> float:
    """
    Calculate Recall@K: fraction of ground truth top-K results found in predictions.
    
    Args:
        predicted_ids: List of IDs returned by the approximate search (top-K)
        ground_truth_ids: List of IDs from exact search (top-K ground truth)
        k: The K value for Recall@K
    
    Returns:
        Recall@K score between 0.0 and 1.0
    """
    if not ground_truth_ids or not predicted_ids:
        return 0.0
    
    # Take top-K from both lists
    ground_truth_set = set(ground_truth_ids[:k])
    predicted_set = set(predicted_ids[:k])
    
    # Count how many ground truth items are in predictions
    intersection = ground_truth_set & predicted_set
    
    if len(ground_truth_set) == 0:
        return 0.0
    
    return len(intersection) / len(ground_truth_set)


def compute_recall_metrics(
    all_predictions: List[List[Tuple[int, str, float]]],
    all_ground_truth: List[List[Tuple[int, str, float]]],
    k: int = 5
) -> Dict[str, float]:
    """
    Compute average Recall@K across multiple queries.
    
    Args:
        all_predictions: List of search results for each query [(id, text, score), ...]
        all_ground_truth: List of ground truth results for each query [(id, text, score), ...]
        k: The K value for Recall@K
    
    Returns:
        Dictionary with recall metrics
    """
    if len(all_predictions) != len(all_ground_truth):
        raise ValueError("Number of predictions must match number of ground truth queries")
    
    recalls = []
    for pred_results, gt_results in zip(all_predictions, all_ground_truth):
        pred_ids = [r[0] for r in pred_results]
        gt_ids = [r[0] for r in gt_results]
        rec = recall_at_k(pred_ids, gt_ids, k)
        recalls.append(rec)
    
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    return {
        "recall_at_k": avg_recall,
        "recall_std": math.sqrt(sum((r - avg_recall) ** 2 for r in recalls) / len(recalls)) if len(recalls) > 1 else 0.0,
        "n_queries": len(recalls),
    }


def measure_throughput(
    search_func,
    query_vectors: List,
    top_k: int = 5,
    duration_seconds: float = 5.0
) -> Dict[str, float]:
    """
    Measure query throughput (queries per second) by running queries continuously.
    
    Args:
        search_func: Function that takes (query_vector, top_k) and returns results
        query_vectors: List of query vectors to use
        top_k: Number of results to retrieve
        duration_seconds: How long to run the throughput test
    
    Returns:
        Dictionary with throughput metrics
    """
    if query_vectors is None:
        return {"qps": 0.0, "total_queries": 0, "duration_s": 0.0}
    try:
        n_queries = len(query_vectors)
    except TypeError:
        return {"qps": 0.0, "total_queries": 0, "duration_s": 0.0}
    if n_queries == 0:
        return {"qps": 0.0, "total_queries": 0, "duration_s": 0.0}
    
    query_count = 0
    start_time = time.perf_counter()
    end_time = start_time + duration_seconds
    
    # Cycle through query vectors
    query_idx = 0
    
    while time.perf_counter() < end_time:
        search_func(query_vectors[query_idx], top_k=top_k)
        query_count += 1
        query_idx = (query_idx + 1) % n_queries
    
    elapsed = time.perf_counter() - start_time
    qps = query_count / elapsed if elapsed > 0 else 0.0
    
    return {
        "qps": qps,
        "total_queries": query_count,
        "duration_s": elapsed,
    }


def get_memory_usage_mb() -> float:
    """
    Get current process memory usage in MB.
    Requires psutil package.
    """
    if not PSUTIL_AVAILABLE:
        return float("nan")
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB
    except Exception:
        return float("nan")


def get_disk_usage_mb(path: str) -> float:
    """
    Get disk usage of a directory in MB.
    
    Args:
        path: Directory path to check
    
    Returns:
        Size in MB, or NaN if path doesn't exist or can't be accessed
    """
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
        return total_size / (1024 * 1024)  # Convert to MB
    except Exception:
        return float("nan")
