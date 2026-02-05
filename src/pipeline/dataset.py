import os
import random
from typing import List, Dict, Iterable, Optional, Tuple


BASE_SENTENCES = [
    "Vector databases store embeddings for similarity search.",
    "Large language models often use retrieval augmented generation.",
    "PostgreSQL can be extended with pgvector for vector operations.",
    "Qdrant is designed for efficient vector search and filtering.",
    "Milvus is a scalable vector database for large datasets.",
    "FAISS is a library for approximate nearest neighbor search.",
    "Embeddings represent text as dense numerical vectors.",
    "Cosine similarity is commonly used for semantic search.",
    "Docker helps ensure reproducible development environments.",
    "Benchmarking measures insertion time and query latency.",
    "Indexing strategies affect performance and accuracy.",
    "Filters and payloads are useful for metadata search.",
    "Cloud deployments introduce network latency and scaling factors.",
    "Batching inserts can significantly improve ingestion speed.",
    "HNSW is a popular index structure for vector search.",
]


TOPICS = [
    "sports", "finance", "medicine", "databases", "ai", "music",
    "movies", "books", "travel", "security", "networks", "cloud"
]


def load_sample_dataset() -> List[Dict]:
    texts = [
        "What is a vector database?",
        "How do large language models work?",
        "PostgreSQL supports extensions like pgvector.",
        "Qdrant is a vector search engine written in Rust.",
        "FAISS is used for efficient similarity search.",
        "Milvus is a distributed vector database.",
        "Embeddings represent text as numerical vectors.",
        "Cosine similarity is commonly used in vector search.",
        "Docker helps create reproducible environments.",
        "Retrieval-Augmented Generation improves LLM accuracy.",
    ]
    return [{"id": i, "text": t, "topic": "ai"} for i, t in enumerate(texts)]


def load_synthetic_dataset(n: int = 1000, seed: int = 42) -> List[Dict]:
    """
    Generates a repeatable synthetic dataset of size n.
    Useful to scale benchmarks without external downloads.
    """
    random.seed(seed)
    data = []
    for i in range(n):
        topic = random.choice(TOPICS)
        s1 = random.choice(BASE_SENTENCES)
        s2 = random.choice(BASE_SENTENCES)
        s3 = random.choice(BASE_SENTENCES)
        text = f"[{topic}] {s1} {s2} {s3} (doc_id={i})"
        data.append({"id": i, "text": text, "topic": topic})
    return data


def _normalize_exts(exts: Iterable[str]) -> List[str]:
    normalized = []
    for ext in exts:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        normalized.append(ext)
    return normalized


def load_file_dataset(
    root_dir: str,
    exts: Iterable[str] = None,
    max_docs: int = 200,
    min_chars: int = 50,
    max_chars: int = 1000,
) -> List[Dict]:
    """
    Load a small real dataset from local files.
    Each file becomes one document with its text content.
    """
    if exts is None:
        exts = ["txt", "md", "markdown", "rst", "json", "jsonl", "csv"]
    allowed_exts = set(_normalize_exts(exts))

    data: List[Dict] = []
    doc_id = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() not in allowed_exts:
                continue

            path = os.path.join(dirpath, filename)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
            except OSError:
                continue

            if len(text) < min_chars:
                continue
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]

            topic = ext.lstrip(".") or "file"
            data.append({"id": doc_id, "text": text, "topic": topic})
            doc_id += 1

            if max_docs and len(data) >= max_docs:
                return data

    return data


def _iter_hdf5_datasets(h5):
    datasets = []

    def _visit(name, obj):
        try:
            import h5py
        except Exception:
            return
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    h5.visititems(_visit)
    return datasets


def _iter_hdf5_vector_datasets(h5) -> List[str]:
    datasets = []

    def _visit(name, obj):
        try:
            import h5py
        except Exception:
            return
        if isinstance(obj, h5py.Dataset):
            if getattr(obj, "ndim", 0) == 2 and getattr(obj.dtype, "kind", "") == "f":
                datasets.append(name)

    h5.visititems(_visit)
    return datasets


def _decode_text(value) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return value.decode(errors="ignore")
    return str(value)


def _choose_text_dataset(h5, text_key: Optional[str]) -> str:
    if text_key:
        if text_key in h5:
            return text_key
        raise ValueError(f"HDF5 text dataset '{text_key}' not found.")

    dataset_paths = _iter_hdf5_datasets(h5)
    if not dataset_paths:
        raise ValueError("No datasets found in HDF5 file.")

    preferred_names = {
        "text", "texts", "sentence", "sentences", "review", "reviews",
        "content", "document", "documents", "data",
    }
    for path in dataset_paths:
        name = path.split("/")[-1].lower()
        if name in preferred_names:
            return path

    # Default to the first dataset if nothing matches common names.
    return dataset_paths[0]


def load_hdf5_dataset(
    path: str,
    text_key: Optional[str] = None,
    max_docs: int = 200,
    min_chars: int = 50,
    max_chars: int = 1000,
) -> List[Dict]:
    """
    Load a dataset from an HDF5 file.

    The HDF5 file should contain a dataset of strings/bytes (one text per row),
    or a compound dataset with a text field. Use text_key to specify which
    dataset or field to use.
    """
    try:
        import h5py
    except Exception as exc:
        raise ImportError(
            "h5py is required to load HDF5 datasets. Install with: pip install h5py"
        ) from exc

    data: List[Dict] = []
    with h5py.File(path, "r") as h5:
        dataset_path = _choose_text_dataset(h5, text_key)
        ds = h5[dataset_path]

        # If compound dtype, try to find a text field.
        text_field = None
        if hasattr(ds.dtype, "names") and ds.dtype.names:
            if text_key and text_key in ds.dtype.names:
                text_field = text_key
            else:
                for candidate in ds.dtype.names:
                    if candidate.lower() in {"text", "sentence", "review", "content"}:
                        text_field = candidate
                        break
            if text_field is None:
                raise ValueError(
                    "HDF5 dataset has multiple fields. "
                    "Pass a text field name via --dataset_hdf5_text_key."
                )

        doc_id = 0
        for row in ds:
            if max_docs and len(data) >= max_docs:
                break
            value = row[text_field] if text_field else row
            text = _decode_text(value).strip()
            if len(text) < min_chars:
                continue
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            data.append({"id": doc_id, "text": text, "topic": "hdf5"})
            doc_id += 1

    return data


def load_hdf5_vectors(
    path: str,
    vectors_key: Optional[str] = None,
    query_key: Optional[str] = None,
    max_docs: int = 200,
    max_queries: int = 5,
) -> Tuple[List[Dict], List, int]:
    """
    Load precomputed embeddings from an HDF5 file.

    The HDF5 file should contain 2D float datasets (n, dim).
    Optionally specify vectors_key and query_key to pick datasets explicitly.
    Returns rows (with embeddings), query_vectors, and embedding dimension.
    """
    try:
        import h5py
    except Exception as exc:
        raise ImportError(
            "h5py is required to load HDF5 datasets. Install with: pip install h5py"
        ) from exc

    with h5py.File(path, "r") as h5:
        vector_paths = _iter_hdf5_vector_datasets(h5)
        if not vector_paths:
            raise ValueError("No 2D float datasets found in HDF5 file.")

        # Pick vectors dataset
        if vectors_key:
            if vectors_key not in h5:
                raise ValueError(f"HDF5 vectors dataset '{vectors_key}' not found.")
            vectors_ds = h5[vectors_key]
        else:
            if "train" in h5:
                vectors_ds = h5["train"]
            else:
                vectors_ds = max(
                    (h5[p] for p in vector_paths),
                    key=lambda ds: ds.shape[0],
                )

        # Pick query dataset
        if query_key:
            if query_key not in h5:
                raise ValueError(f"HDF5 query dataset '{query_key}' not found.")
            query_ds = h5[query_key]
        else:
            if "test" in h5 and "test" in vector_paths:
                query_ds = h5["test"]
            else:
                query_ds = vectors_ds

        # Slice data
        n_docs = max_docs if max_docs else vectors_ds.shape[0]
        n_queries = max_queries if max_queries else min(5, query_ds.shape[0])

        vectors = vectors_ds[:n_docs]
        query_vectors = query_ds[:n_queries]

        rows: List[Dict] = []
        for i, vec in enumerate(vectors):
            rows.append({"id": i, "text": f"hdf5_vec_{i}", "embedding": vec})

        dim = vectors.shape[1] if vectors.ndim == 2 else len(vectors[0])
        return rows, query_vectors, dim
