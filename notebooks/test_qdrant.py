from src.pipeline.dataset import load_sample_dataset
from src.pipeline.embed import EmbeddingGenerator
from src.backends.qdrant_backend import QdrantBackend


def main():
    data = load_sample_dataset()
    texts = [d["text"] for d in data]

    embedder = EmbeddingGenerator()
    vectors = embedder.embed(texts)

    rows = []
    for item, vec in zip(data, vectors):
        rows.append({"id": item["id"], "text": item["text"], "embedding": vec})

    qd = QdrantBackend(host="127.0.0.1", port=6333, collection_name="documents")
    qd.recreate_collection(dim=embedder.dim)
    qd.upsert(rows)

    print("Inserted rows:", len(rows))

    query_text = "What are embeddings used for?"
    query_vec = embedder.embed([query_text])[0]

    results = qd.search(query_vec, top_k=5)
    print("\nQuery:", query_text)
    print("Top results (Qdrant score: higher is better):")
    for _id, text, score in results:
        print(f"  id={_id:02d} score={score:.4f} text={text}")

    bench = qd.benchmark_search(query_vec, top_k=5, repeats=50)
    print("\nLatency benchmark (same query repeated):", bench)


if __name__ == "__main__":
    main()