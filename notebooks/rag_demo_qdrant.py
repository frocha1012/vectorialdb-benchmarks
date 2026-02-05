import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import requests

from src.pipeline.dataset import load_synthetic_dataset
from src.pipeline.embed import EmbeddingGenerator
from src.backends.qdrant_backend import QdrantBackend


CACHE_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_META = CACHE_DIR / "rag_demo_meta.json"
CACHE_EMB = CACHE_DIR / "rag_demo_embeddings.npy"


def build_or_load_corpus(n_docs: int, embedder: EmbeddingGenerator) -> List[Dict[str, Any]]:
    """
    Builds a synthetic corpus and caches embeddings to disk.
    This avoids recomputing embeddings every time you demo RAG.
    """
    if CACHE_META.exists() and CACHE_EMB.exists():
        meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
        if meta.get("n_docs") == n_docs and meta.get("dim") == embedder.dim and meta.get("model") == embedder.model_name:
            print(f"[cache] Loading cached corpus: {n_docs} docs")
            items = meta["items"]
            embeddings = np.load(CACHE_EMB)
            # stitch back
            for i, it in enumerate(items):
                it["embedding"] = embeddings[i]
            return items

    print(f"[build] Generating corpus + embeddings for n_docs={n_docs} (first run will take time)")
    data = load_synthetic_dataset(n=n_docs, seed=42)
    texts = [d["text"] for d in data]
    vectors = embedder.embed(texts)

    items = []
    for item, vec in zip(data, vectors):
        items.append(
            {
                "id": int(item["id"]),
                "text": item["text"],
                "topic": item.get("topic", ""),
                "embedding": vec,
            }
        )

    # cache
    embeddings = np.array([it["embedding"] for it in items], dtype=np.float32)
    meta = {
        "n_docs": n_docs,
        "dim": embedder.dim,
        "model": embedder.model_name,
        "items": [{"id": it["id"], "text": it["text"], "topic": it["topic"]} for it in items],
    }
    CACHE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(CACHE_EMB, embeddings)
    print("[cache] Saved embeddings to data/ for faster reruns")

    return items


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Simple RAG prompt that cites sources.
    """
    context_block = "\n\n".join(
        [f"[Source {i+1} | topic={c.get('topic','')}] {c['text']}" for i, c in enumerate(contexts)]
    )

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the sources below.
If the sources are insufficient, say you don't know.

Question:
{question}

Sources:
{context_block}

Answer (include short citations like [1], [2] where relevant):
"""
    return prompt


def generate_with_ollama(prompt: str, model: str = "llama3.1:8b") -> Optional[str]:
    """
    Optional: if you have Ollama running locally.
    - Install: https://ollama.com
    - Run: ollama serve
    - Pull model: ollama pull llama3.1:8b
    """
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("response")
    except Exception:
        return None


def main():
    # 1) Config
    n_docs = int(os.environ.get("RAG_N_DOCS", "5000"))  # change to 10000 if you want
    top_k = int(os.environ.get("RAG_TOP_K", "5"))
    use_ollama = os.environ.get("RAG_USE_OLLAMA", "0") == "1"
    ollama_model = os.environ.get("RAG_OLLAMA_MODEL", "llama3.1:8b")

    # 2) Embedder
    embedder = EmbeddingGenerator()
    # Small tweak: store model_name for cache metadata
    embedder.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # 3) Build/load corpus
    items = build_or_load_corpus(n_docs, embedder)

    # 4) Load into Qdrant (fresh collection for demo)
    qd = QdrantBackend(host="127.0.0.1", port=6333, collection_name="rag_demo")
    qd.recreate_collection(dim=embedder.dim)
    qd.upsert(items, batch_size=256)
    print(f"[qdrant] Loaded {len(items)} documents into collection 'rag_demo'")

    # 5) Interactive loop
    print("\n--- Tiny RAG Demo (Qdrant) ---")
    print("Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("Question> ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        qvec = embedder.embed([question])[0]
        hits = qd.search(qvec, top_k=top_k)

        # Convert hits into context objects (include topic if present)
        contexts = []
        for _id, text, score in hits:
            # We don't fetch topic back from qdrant in your current backend, so we approximate from stored items:
            topic = items[_id].get("topic") if _id < len(items) else ""
            contexts.append({"id": _id, "text": text, "topic": topic, "score": score})

        print("\nTop retrieved sources:")
        for i, c in enumerate(contexts, 1):
            print(f"  {i}. id={c['id']} score={c['score']:.4f} topic={c['topic']}")

        prompt = build_prompt(question, contexts)

        print("\n--- RAG Prompt (what gets sent to the generator) ---")
        print(prompt)

        if use_ollama:
            print(f"\n[ollama] Generating with model '{ollama_model}' ...")
            answer = generate_with_ollama(prompt, model=ollama_model)
            if answer:
                print("\n--- Generated Answer ---")
                print(answer.strip())
            else:
                print("\n[ollama] Could not generate (is Ollama running on localhost:11434?)")
        else:
            # Tiny non-LLM fallback: “answer” by stitching the best source + citations
            print("\n--- Tiny Non-LLM Answer (fallback) ---")
            if contexts:
                print(f"Based on the retrieved documents, the most relevant information is: {contexts[0]['text']} [1]")
                if len(contexts) > 1:
                    print(f"Additional related context: {contexts[1]['text']} [2]")
            else:
                print("I don't know. The sources retrieved were insufficient.")

        print("\n")


if __name__ == "__main__":
    main()
