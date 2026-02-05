"""
Export a small IMDb dataset subset into individual text files.
"""
import argparse
import os
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_docs", type=int, default=1000, help="Number of docs to export")
    args = parser.parse_args()

    out_dir = os.path.join("data", "imdb_docs")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("imdb", split="train")
    max_docs = args.max_docs

    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        text = row.get("text", "").strip()
        if not text:
            continue
        path = os.path.join(out_dir, f"imdb_{i:04d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Wrote {min(max_docs, len(ds))} docs to {out_dir}")


if __name__ == "__main__":
    main()
