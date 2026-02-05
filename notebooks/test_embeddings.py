from src.pipeline.dataset import load_sample_dataset
from src.pipeline.embed import EmbeddingGenerator

data = load_sample_dataset()
texts = [d["text"] for d in data]

embedder = EmbeddingGenerator()
vectors = embedder.embed(texts)

print("Embedding dimension:", embedder.dim)
print("Number of vectors:", len(vectors))
print("First vector (first 5 values):", vectors[0][:5])
