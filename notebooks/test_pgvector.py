from src.pipeline.dataset import load_sample_dataset
from src.pipeline.embed import EmbeddingGenerator
from src.backends.pgvector_backend import PgVectorBackend

def main():
    data = load_sample_dataset()
    texts = [d["text"] for d in data]

    embedder = EmbeddingGenerator()
    vectors = embedder.embed(texts)

    # Build rows to insert
    rows = []
    for item, vec in zip(data, vectors):
        rows.append({
            "id": item["id"],
            "text": item["text"],
            "embedding": vec,  # numpy array ok with pgvector adapter
        })

    pg = PgVectorBackend(host="127.0.0.1", port=5433, password="postgres", user="postgres", dbname="vectordb")
    pg.create_schema(dim=embedder.dim)
    pg.clear()
    pg.upsert(rows)

    print("Inserted rows:", len(rows))

    # Query with one of the sentences
    query_text = "What are embeddings used for?"
    query_vec = embedder.embed([query_text])[0]

    results = pg.search(query_vec, top_k=5)
    print("\nQuery:", query_text)
    print("Top results:")
    for _id, text, dist in results:
        print(f"  id={_id:02d} dist={dist:.4f} text={text}")

    bench = pg.benchmark_search(query_vec, top_k=5, repeats=50)
    print("\nLatency benchmark (same query repeated):", bench)

if __name__ == "__main__":
    main()
