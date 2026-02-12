import os

# Avoid Mac OpenMP crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from tqdm import tqdm
import math

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

TOP_K = 20
MAX_USERS = 5000   # evaluation sample (safe on CPU)

DEVICE = "cpu"

# ---------- Metrics ----------

def recall_at_k(preds, targets, k):
    preds_k = preds[:k]
    hit_count = len(set(preds_k) & set(targets))
    return hit_count / len(targets) if len(targets) > 0 else 0


def ndcg_at_k(preds, targets, k):
    dcg = 0.0
    for i, item in enumerate(preds[:k]):
        if item in targets:
            dcg += 1 / math.log2(i + 2)

    ideal_hits = min(len(targets), k)
    idcg = sum([1 / math.log2(i + 2) for i in range(ideal_hits)])

    return dcg / idcg if idcg > 0 else 0


# ---------- Load Pipeline ----------

class FeedEvaluator:

    def __init__(self):
        print("Loading artifacts...")

        self.user_emb = np.load(MODEL_DIR / "user_embeddings.npy")
        self.item_emb = np.load(MODEL_DIR / "item_embeddings.npy")

        self.index = faiss.read_index(str(MODEL_DIR / "faiss_item.index"))

        self.ranking_model = self._load_ranker()

        faiss.normalize_L2(self.item_emb)

    def _load_ranker(self):
        from train_ranker import RankingModel

        model = RankingModel(
            user_emb=self.user_emb,
            item_emb=self.item_emb
        )

        model.load_state_dict(torch.load(MODEL_DIR / "ranking_model.pt", map_location=DEVICE))
        model.eval()

        return model

    def recommend(self, user_id, retrieval_k=100):

        user_vec = self.user_emb[user_id:user_id+1]
        faiss.normalize_L2(user_vec)

        scores, candidates = self.index.search(user_vec, retrieval_k)

        candidate_items = candidates[0]
        sim_scores = scores[0]

        users = torch.tensor([user_id] * len(candidate_items))
        items = torch.tensor(candidate_items)
        sim_tensor = torch.tensor(sim_scores, dtype=torch.float32)

        popularity = torch.zeros(len(candidate_items))

        with torch.no_grad():
            preds = self.ranking_model(
                users,
                items,
                sim_tensor,
                popularity
            )

        ranked_idx = torch.argsort(preds, descending=True)

        final_items = candidate_items[ranked_idx[:TOP_K]]

        return final_items.tolist()


# ---------- Evaluation Loop ----------

def main():
    print("Loading test set...")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    print("Preparing ground truth...")
    user_truth = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    users = list(user_truth.keys())[:MAX_USERS]

    evaluator = FeedEvaluator()

    recalls = []
    ndcgs = []

    print("Evaluating recommender...")

    for user_id in tqdm(users):
        gt_items = user_truth[user_id]

        preds = evaluator.recommend(user_id)

        recalls.append(recall_at_k(preds, gt_items, TOP_K))
        ndcgs.append(ndcg_at_k(preds, gt_items, TOP_K))

    print("\n===== FINAL RESULTS =====")
    print(f"Users evaluated: {len(users)}")
    print(f"Recall@{TOP_K}: {np.mean(recalls):.4f}")
    print(f"NDCG@{TOP_K}: {np.mean(ndcgs):.4f}")


if __name__ == "__main__":
    main()
