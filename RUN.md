# How to Run

This project benchmarks vector databases (FAISS, Milvus, pgvector, Qdrant, MongoDB Atlas).

## 0) Python environment (recommended)

This project currently needs Python **3.12** (3.13 has limited wheels for pandas/FAISS).

Create and activate a venv:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install pandas matplotlib sentence-transformers tqdm h5py faiss-cpu python-dotenv \
  psycopg2-binary qdrant-client pymilvus pymongo psutil
```

Note: on Windows, FAISS may require conda if `faiss-cpu` is not available.

## 1) Start databases (Docker)

From the repo root:

```powershell
docker-compose up -d
```

This brings up pgvector (Postgres), Qdrant, and MongoDB (local container).

To run Milvus:

```powershell
docker-compose -f docker-compose.milvus.yml up -d
```

## 2) Optional: MongoDB Atlas

If you want Atlas benchmarks, set the connection string:

```powershell
$env:MONGODB_ATLAS_URI="your-connection-string"
```

Otherwise, Atlas is skipped automatically.

## 3) Benchmark types (what you can run)

You can run these benchmark variants:

- **Single-size synthetic**: builds `n` documents by sampling topics and sentences
  from `src/pipeline/dataset.py::load_synthetic_dataset()`. Good for quick,
  repeatable runs.
- **Single-size real dataset**: reads text files from a folder via
  `src/pipeline/dataset.py::load_file_dataset()`. Each file becomes one document,
  with filtering by length and extension.
- **Single-size HDF5 dataset**: reads a text dataset from an `.hdf5` file via
  `src/pipeline/dataset.py::load_hdf5_dataset()`. Use `--dataset_hdf5_text_key`
  if the text dataset/field is not auto-detected.
- **Multi-size suite**: repeats the same benchmark across multiple dataset sizes
  to show scaling trends.
- **MongoDB Atlas**: optional add-on if `MONGODB_ATLAS_URI` is set.

All variants benchmark the same backends (FAISS, Milvus, pgvector, Qdrant; Atlas optional).

## 4) Run a single benchmark

Synthetic dataset (default n=1000):

```powershell
python -m src.benchmark.run_benchmark
```

Custom dataset folder:

```powershell
python -m src.benchmark.run_benchmark --dataset_dir "data" --dataset_max_docs 200
```

HDF5 dataset file:

```powershell
python -m src.benchmark.run_benchmark --dataset_hdf5 "data\agnews-dataset.hdf5" --dataset_max_docs 200
```

If the text dataset is not auto-detected, provide the dataset or field name:

```powershell
python -m src.benchmark.run_benchmark --dataset_hdf5 "data\agnews-dataset.hdf5" --dataset_hdf5_text_key "text"
```

Note: HDF5 loading requires `h5py` (`pip install h5py`).

### HDF5 with precomputed embeddings (AG News)

If your HDF5 file contains embeddings (2D float arrays) instead of text, you can
run in vector mode. For `agnews-dataset.hdf5`, the datasets are `train` (vectors)
and `test` (queries), so run:

```powershell
python -m src.benchmark.run_benchmark --dataset_hdf5 "data\agnews-dataset.hdf5" ^
  --dataset_hdf5_vectors_key "train" --dataset_hdf5_query_key "test" ^
  --dataset_max_docs 200 --dataset_hdf5_max_queries 5
```

Other options:
- `--n 2000`
- `--dataset_min_chars 50`
- `--dataset_max_chars 1000`
- `--dataset_exts "txt,md,markdown,rst,json,jsonl,csv"`

Outputs go to `results/` and a CSV is copied to `site/public/benchmark_results.csv`.

### How results are analyzed (single benchmark)

Each single run reports:
- **Insert time**: total ms to insert all vectors.
- **Latency**: avg + p50/p95/p99 per-query latency (ms).
- **Recall@K**: compares each backend to FAISS exact search.
- **Throughput**: QPS measured by running queries for ~3 seconds.
- **Memory**: process memory delta (requires `psutil`).

Use single runs to compare backends on a fixed dataset.

### Using the built-in IMDb files

The `data/imdb_docs` folder contains movie reviews as plain `.txt` files.
These were generated with:

```powershell
python -m notebooks.export_imdb_dataset --max_docs 1000
```

That script downloads the IMDb dataset (via `datasets`), writes each review to
`data/imdb_docs/imdb_XXXX.txt`, and you can benchmark it like:

```powershell
python -m src.benchmark.run_benchmark --dataset_dir "data\imdb_docs" --dataset_max_docs 200
```

## 5) Run a suite (multiple sizes)

```powershell
python -m src.benchmark.run_suite --sizes "1000,2000,3000,4000" --topk 5
```

Outputs go to `results/` and a CSV is copied to `site/public/benchmark_results.csv`.

### How results are analyzed (suite)

The suite repeats the same metrics across multiple dataset sizes and generates
scaling plots for latency, insert time, throughput, accuracy, and memory.

## 6) View the web dashboard

From `site/`:

```powershell
.\install_webpage.bat
.\start_webpage.bat
```

Then open the local URL shown in the terminal.
