import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUT_PATH = DATA_DIR / "ranking_train.parquet"

MAX_USERS = 20000
CANDIDATES_PER_USER = 100

def main():
    print("Loading processed data...")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")

    print("Loading embeddings...")
    user_emb = np.load(MODEL_DIR / "user_embeddings.npy")
    item_emb = np.load(MODEL_DIR / "item_embeddings.npy")

    print("Loading FAISS index...")
    index = faiss.read_index(str(MODEL_DIR / "faiss_item.index"))

    # Normalize item embeddings (important for cosine similarity)
    faiss.normalize_L2(item_emb)

    print("Preparing user interaction dictionary...")
    user_groups = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    users = list(user_groups.keys())[:MAX_USERS]

    rows = []

    print("Building ranking training samples...")

    for user_id in tqdm(users):
        u_emb = user_emb[user_id:user_id+1]
        faiss.normalize_L2(u_emb)

        scores, candidates = index.search(u_emb, CANDIDATES_PER_USER)

        pos_items = user_groups[user_id]

        for rank, item_id in enumerate(candidates[0]):
            label = 1 if item_id in pos_items else 0

            sim_score = scores[0][rank]

            popularity = len(train_df[train_df["item_idx"] == item_id])  

            rows.append([
                user_id,
                item_id,
                sim_score,
                popularity,
                label
            ])

    ranking_df = pd.DataFrame(rows, columns=[
        "user_idx",
        "item_idx",
        "sim_score",
        "popularity",
        "label"
    ])

    print("Saving ranking dataset...")
    ranking_df.to_parquet(OUT_PATH)

    print("Ranking dataset created successfully.")
    print("Total samples:", len(ranking_df))

if __name__ == "__main__":
    main()
