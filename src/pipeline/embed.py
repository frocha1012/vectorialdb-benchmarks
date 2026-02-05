from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            vec = self.model.encode(text, normalize_embeddings=True)
            embeddings.append(vec)
        return embeddings
