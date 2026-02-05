# Benchmark Metrics Documentation

This document explains all metrics collected during vector database benchmarking.

---

## Table of Contents

1. [Insertion Metrics](#insertion-metrics)
2. [Query Latency Metrics](#query-latency-metrics)
3. [Search Accuracy Metrics](#search-accuracy-metrics)
4. [Throughput Metrics](#throughput-metrics)
5. [Resource Efficiency Metrics](#resource-efficiency-metrics)
6. [Dataset Metadata](#dataset-metadata)

---

## Insertion Metrics

### `insert_ms`
- **Type**: Float (milliseconds)
- **Description**: Total time taken to insert all vectors into the database
- **Measurement**: Time from start of insertion to completion of all vector inserts
- **Interpretation**: Lower is better. Measures how quickly the database can ingest data.
- **Example**: `276.95` means it took 276.95 milliseconds to insert all vectors

### `insert_count`
- **Type**: Integer
- **Description**: Number of vectors that were inserted
- **Measurement**: Count of vectors in the dataset
- **Interpretation**: Should match the dataset size (`n_dataset`). Used to verify all vectors were inserted.
- **Example**: `1000` means 1,000 vectors were inserted

---

## Query Latency Metrics

### `avg_ms`
- **Type**: Float (milliseconds)
- **Description**: Average query latency across all test queries
- **Measurement**: Mean of all individual query latencies
- **Interpretation**: Lower is better. Represents typical query performance.
- **Example**: `16.69` means average query took 16.69 milliseconds

### `p50_ms`
- **Type**: Float (milliseconds)
- **Description**: 50th percentile latency (median)
- **Measurement**: Half of queries are faster, half are slower than this value
- **Interpretation**: Lower is better. More robust than average as it's not affected by outliers.
- **Example**: `20.75` means 50% of queries took ≤ 20.75 milliseconds

### `p95_ms`
- **Type**: Float (milliseconds)
- **Description**: 95th percentile latency
- **Measurement**: 95% of queries are at or below this latency
- **Interpretation**: Lower is better. Measures tail latency - important for user experience.
- **Example**: `22.56` means 95% of queries took ≤ 22.56 milliseconds

### `p99_ms`
- **Type**: Float (milliseconds)
- **Description**: 99th percentile latency
- **Measurement**: 99% of queries are at or below this latency
- **Interpretation**: Lower is better. Measures worst-case performance (excluding extreme outliers).
- **Example**: `22.79` means 99% of queries took ≤ 22.79 milliseconds

### `n`
- **Type**: Integer
- **Description**: Number of queries executed for latency measurement
- **Measurement**: Count of queries used to compute latency statistics
- **Interpretation**: Higher sample size = more reliable statistics. Typically 5 queries per benchmark.
- **Example**: `5` means 5 queries were measured

---

## Search Accuracy Metrics

### `recall_at_k`
- **Type**: Float (0.0 to 1.0)
- **Description**: Average Recall@K score across all test queries
- **Measurement**: Fraction of true top-K results found by the approximate search, compared to exact search (ground truth)
- **Calculation**: `(number of ground truth items found) / (total ground truth items)`
- **Interpretation**: 
  - `1.0` = Perfect accuracy (all correct results found)
  - `0.8` = 80% of correct results found
  - `0.0` = No correct results found
- **Example**: `1.0` means perfect recall - all correct results were retrieved

### `recall_std`
- **Type**: Float
- **Description**: Standard deviation of Recall@K across queries
- **Measurement**: Variability in recall scores between different queries
- **Interpretation**: 
  - Lower is better (more consistent accuracy)
  - `0.0` = Perfect consistency across queries
  - Higher values = Some queries perform better/worse than others
- **Example**: `0.0` means all queries had identical recall scores

### `n_queries`
- **Type**: Integer
- **Description**: Number of queries used for recall calculation
- **Measurement**: Count of test queries evaluated
- **Interpretation**: Higher sample size = more reliable accuracy assessment. Typically 5 queries.
- **Example**: `5` means recall was calculated using 5 different queries

---

## Throughput Metrics

### `qps`
- **Type**: Float (queries per second)
- **Description**: Query throughput - number of queries the database can handle per second
- **Measurement**: Continuous query execution for a fixed duration (default: 3 seconds), then `total_queries / duration`
- **Interpretation**: Higher is better. Measures sustained query performance under load.
- **Example**: `34.76` means the database can handle ~35 queries per second

### `total_queries`
- **Type**: Integer
- **Description**: Total number of queries executed during throughput test
- **Measurement**: Count of queries run in the throughput measurement period
- **Interpretation**: Used to calculate QPS. Higher values = more reliable throughput measurement.
- **Example**: `105` means 105 queries were executed during the test

### `duration_s`
- **Type**: Float (seconds)
- **Description**: Actual duration of the throughput test
- **Measurement**: Time elapsed during continuous query execution
- **Interpretation**: Should be close to the target duration (default: 3.0 seconds). Used to calculate QPS.
- **Example**: `3.02` means the test ran for 3.02 seconds

---

## Resource Efficiency Metrics

### `memory_mb`
- **Type**: Float (megabytes)
- **Description**: Memory usage delta - change in memory consumption after inserting vectors
- **Measurement**: `(memory after insertion) - (memory before insertion)`
- **Interpretation**: 
  - Lower is better (more memory efficient)
  - Measures how much additional memory the database uses to store vectors
  - May be `NaN` if `psutil` package is not installed
- **Example**: `2.70` means the database used an additional 2.70 MB of memory

---

## Dataset Metadata

### `n_dataset`
- **Type**: Integer
- **Description**: Dataset size - number of vectors in the test dataset
- **Measurement**: Size of the synthetic dataset generated for benchmarking
- **Interpretation**: Used to analyze how performance scales with dataset size. Larger datasets test scalability.
- **Example**: `1000` means the benchmark used 1,000 vectors

### `db`
- **Type**: String
- **Description**: Database name/identifier
- **Values**: 
  - `"faiss"` - FAISS (in-memory baseline)
  - `"milvus"` - Milvus vector database
  - `"pgvector"` - PostgreSQL with pgvector extension
  - `"qdrant"` - Qdrant vector database
- **Interpretation**: Identifies which database system the metrics belong to
- **Example**: `"pgvector"` means these metrics are for PostgreSQL + pgvector

---

## Metric Relationships

### Latency Percentiles
- **p50 ≤ avg ≤ p95 ≤ p99** (typically)
- Large gap between p50 and p95 indicates inconsistent performance
- Small gap indicates consistent, predictable latency

### Throughput vs Latency
- **QPS ≈ 1000 / avg_ms** (approximately)
- Higher QPS usually correlates with lower latency
- But not always - some databases optimize for one over the other

### Recall vs Performance
- Higher recall (accuracy) may come at the cost of lower throughput
- Exact search (FAISS IndexFlatIP) has perfect recall but may be slower
- Approximate search (ANN) trades some accuracy for speed

### Memory vs Dataset Size
- Memory usage typically scales with dataset size
- Different databases have different memory efficiency
- Some databases use more memory for better performance

---

## How to Interpret Results

### Best Performance
- **Low latency**: p50_ms, p95_ms, p99_ms all low
- **High throughput**: qps high
- **Low memory**: memory_mb low
- **Fast insertion**: insert_ms low
- **High accuracy**: recall_at_k = 1.0

### Trade-offs
- **Speed vs Accuracy**: Some databases prioritize speed (lower latency) over perfect recall
- **Memory vs Speed**: More memory can enable faster queries through caching
- **Insertion vs Query**: Some databases optimize for fast queries at the cost of slower insertion

### Scalability
- Compare metrics across different `n_dataset` values
- Good scalability = metrics don't degrade significantly as dataset grows
- Poor scalability = latency/insertion time increases dramatically with dataset size

---

## Notes

- All latency measurements are in **milliseconds (ms)**
- All memory measurements are in **megabytes (MB)**
- All time measurements use high-precision timers (`time.perf_counter()`)
- Ground truth for recall is computed using FAISS exact search (IndexFlatIP)
- Throughput test runs for 3 seconds by default
- Memory measurement requires `psutil` package (optional)


