import faiss
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")

USER_EMB_PATH = MODEL_DIR / "user_embeddings.npy"
INDEX_PATH = MODEL_DIR / "faiss_item.index"

TOP_K = 10

def main():
    print("Loading embeddings and FAISS index...")

    user_emb = np.load(USER_EMB_PATH)
    index = faiss.read_index(str(INDEX_PATH))

    user_id = 42  # test user

    query = user_emb[user_id:user_id+1]

    faiss.normalize_L2(query)

    scores, item_ids = index.search(query, TOP_K)

    print("Recommended item IDs for user", user_id)
    print(item_ids[0])
    print("Similarity scores:")
    print(scores[0])

if __name__ == "__main__":
    main()
