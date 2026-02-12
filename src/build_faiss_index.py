import faiss
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")

ITEM_EMB_PATH = MODEL_DIR / "item_embeddings.npy"
INDEX_PATH = MODEL_DIR / "faiss_item.index"

EMBED_DIM = 64

def main():
    print("Loading item embeddings...")
    item_emb = np.load(ITEM_EMB_PATH)

    print("Normalizing embeddings...")
    faiss.normalize_L2(item_emb)

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(EMBED_DIM)

    index.add(item_emb)

    print("Total items indexed:", index.ntotal)

    print("Saving index...")
    faiss.write_index(index, str(INDEX_PATH))

    print("FAISS index built successfully.")

if __name__ == "__main__":
    main()
