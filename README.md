# Vector Benchmark Suite

Benchmark multiple vector backends (FAISS, Milvus, pgvector, Qdrant, MongoDB Atlas)
on insertion speed, query latency, throughput, memory, and Recall@K.

## What this project does

- Generates or loads datasets (synthetic, text files, HDF5).
- Embeds text (SentenceTransformer) or uses precomputed embeddings (HDF5).
- Runs a uniform benchmark across backends.
- Outputs CSV/JSON/PNG reports and a simple dashboard.

## Requirements

- Windows + PowerShell
- Python 3.12 (3.13 lacks some wheels)
- Docker (for pgvector, Qdrant, local MongoDB)

## Setup

Create and activate a venv:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install pandas matplotlib sentence-transformers tqdm h5py faiss-cpu python-dotenv \
  psycopg2-binary qdrant-client pymilvus pymongo pgvector psutil
```

Note: on Windows, FAISS may require conda if `faiss-cpu` is not available.

## Start databases (Docker)

From the repo root:

```powershell
docker-compose up -d
```

Milvus:

```powershell
docker-compose -f docker-compose.milvus.yml up -d
```

## Optional: MongoDB Atlas

Set the connection string if you want Atlas results:

```powershell
$env:MONGODB_ATLAS_URI="your-connection-string"
```

Otherwise Atlas is skipped.

## Benchmark types

- **Single-size synthetic**: builds `n` synthetic documents by sampling topics
  and sentences, then embeds them and runs a full benchmark. Best for quick,
  repeatable checks without file I/O.
- **Single-size real dataset (files)**: loads one document per file from a folder
  (e.g., `data/imdb_docs`), applies min/max length filters, then embeds and
  benchmarks. Good for real text distributions.
- **Single-size HDF5 (text)**: reads a text dataset from an `.hdf5` file, embeds
  it, and benchmarks. This is the path for real datasets stored in a local
  database-like HDF5 file.
- **Single-size HDF5 (vectors)**: uses precomputed embeddings stored in `.hdf5`,
  skipping the embedding model. Ideal for large corpora where vectors are
  already generated (e.g., `train`/`test` datasets).
- **Multi-size suite**: repeats the same benchmark across multiple dataset
  sizes to show scaling trends (latency, throughput, insert time, accuracy).

All runs benchmark the same backends (Atlas optional).

## Benchmark details (what each does)

### 1) Single-size synthetic

Creates `n` documents by sampling topics and base sentences in
`src/pipeline/dataset.py::load_synthetic_dataset()`. It uses the embedding model
to generate vectors, then benchmarks inserts and queries. This is the fastest
way to sanity‑check the system because it avoids file I/O.

### 2) Single-size real dataset (files)

Loads one document per file from a folder using
`src/pipeline/dataset.py::load_file_dataset()`. This keeps real‑world text
distribution and length variation (after min/max filtering). Good for a more
realistic pipeline without requiring a custom loader.

### 3) Single-size HDF5 (text)

Reads a dataset of strings from an HDF5 file via
`src/pipeline/dataset.py::load_hdf5_dataset()`. The loader auto‑detects common
dataset names (like `text`, `reviews`, `content`) or you can provide
`--dataset_hdf5_text_key`. This is meant for real datasets stored as a local
database-like file. It then embeds the text and benchmarks normally.

### 4) Single-size HDF5 (vectors)

Uses precomputed embeddings stored in HDF5 (2D float arrays). The loader
`load_hdf5_vectors()` pulls vectors directly and avoids running the embedding
model. You can choose the dataset names with `--dataset_hdf5_vectors_key` and
`--dataset_hdf5_query_key` (e.g., `train` and `test`). This is ideal for large,
pre‑embedded corpora and is the preferred path when your real dataset is stored
in a database-style HDF5 file.

### 5) Multi-size suite

Runs the same benchmark across multiple dataset sizes. It is designed to show
scaling behavior as data grows (latency, throughput, insert time, accuracy).
It also generates a set of comparison charts and dashboards.

## How each benchmark is analyzed

Every run records the same core metrics; below is what each one means and how
it is measured:

- **Insert time**: total time (ms) to load all vectors into a backend. This
  measures ingestion speed for the given dataset size and embedding dimension.
- **Latency (avg/p50/p95/p99)**: per‑query response time in milliseconds. The
  benchmark runs each query once and summarizes the distribution; p50 is the
  median, p95/p99 show tail latency.
- **Recall@K**: accuracy versus FAISS exact search. For each query, results are
  compared to FAISS top‑K; the final value is the average overlap.
- **Throughput (QPS)**: queries per second measured by looping queries for a
  fixed time window (default ~3 seconds).
- **Memory (MB)**: process memory delta before/after insert. Requires `psutil`.

For the **suite**, the same metrics are computed at multiple dataset sizes and
plotted to show scaling trends (latency growth, throughput drop, insertion
time growth, and accuracy stability).

## How to run (single benchmark)

Synthetic dataset:

```powershell
python -m src.benchmark.run_benchmark
```
Uses the synthetic generator with default `--n 1000` documents.

Text files from a folder:

```powershell
python -m src.benchmark.run_benchmark --dataset_dir "data" --dataset_max_docs 200
```
`--dataset_dir` points to a folder of text files; `--dataset_max_docs` caps how
many documents are loaded.

HDF5 (text dataset inside the file):

```powershell
python -m src.benchmark.run_benchmark --dataset_hdf5 "data\agnews-dataset.hdf5" --dataset_max_docs 200
```
`--dataset_hdf5` points to the HDF5 file; `--dataset_max_docs` caps the number
of rows loaded from the selected dataset.

If the text dataset is not auto-detected:

```powershell
python -m src.benchmark.run_benchmark --dataset_hdf5 "data\agnews-dataset.hdf5" --dataset_hdf5_text_key "text"
```
`--dataset_hdf5_text_key` selects the dataset/field name inside the HDF5 file.

### HDF5 with precomputed embeddings (AG News)

If the HDF5 contains vectors instead of text, use vector mode:

```powershell
python -m src.benchmark.run_benchmark --dataset_hdf5 "data\agnews-dataset.hdf5" ^
  --dataset_hdf5_vectors_key "train" --dataset_hdf5_query_key "test" ^
  --dataset_max_docs 200 --dataset_hdf5_max_queries 5
```
`--dataset_hdf5_vectors_key` selects the vectors dataset (documents).  
`--dataset_hdf5_query_key` selects the query vectors dataset.  
`--dataset_hdf5_max_queries` caps the number of query vectors used for latency
and recall (more queries = more stable stats, longer runtime).

## IMDb dataset (optional)

Generate local text files:

```powershell
python -m notebooks.export_imdb_dataset --max_docs 1000
```
`--max_docs` caps how many IMDb reviews are exported to `data/imdb_docs`.

Run:

```powershell
python -m src.benchmark.run_benchmark --dataset_dir "data\imdb_docs" --dataset_max_docs 200
```
Same flags as above; this uses the IMDb file dataset.

## Run a suite (multiple sizes)

```powershell
python -m src.benchmark.run_suite --sizes "1000,2000,3000,4000" --topk 5
```
`--sizes` is a comma‑separated list of dataset sizes to sweep.  
`--topk` sets K for Recall@K and query result size.

## Outputs

Single run outputs:
- `results/run_<timestamp>.json`
- `results/summary_<timestamp>.csv`
- `results/latency_<timestamp>.png`
- `site/public/benchmark_results.csv` (for the dashboard)

Suite outputs:
- `results/suite_<timestamp>.json`
- `results/suite_<timestamp>.csv`
- `results/suite_*_<timestamp>.png`
- `site/public/benchmark_results.csv`

## View the dashboard

From `site/`:

```powershell
.\install_webpage.bat
.\start_webpage.bat
```

Open the URL shown in the terminal.

## Notes on accuracy

- FAISS uses exact search and acts as Recall@K ground truth.
- pgvector uses L2 distance while FAISS uses cosine similarity (normalized IP),
  so Recall@K can be lower unless distance metrics are aligned.

## Related docs

- `RUN.md`: short run guide
- `METRICS.md`: metric definitions
- `ANALYSIS.md`: latest run sanity check

# vectorialdb-benchmarks
