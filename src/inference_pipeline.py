import os

# Fix Mac OpenMP + FAISS + Torch conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import numpy as np
import faiss
from pathlib import Path



MODEL_DIR = Path("models")

EMBED_DIM = 64
TOP_K_RETRIEVAL = 100
TOP_K_FEED = 20

DEVICE = "cpu"

class FeedRecommender:

    def __init__(self):
        print("Loading artifacts...")

        self.user_emb = np.load(MODEL_DIR / "user_embeddings.npy")
        self.item_emb = np.load(MODEL_DIR / "item_embeddings.npy")

        self.index = faiss.read_index(str(MODEL_DIR / "faiss_item.index"))

        self.ranking_model = self._load_ranker()

        faiss.normalize_L2(self.item_emb)

        print("System ready.")

    def _load_ranker(self):
        from train_ranker import RankingModel
        model = RankingModel(
            user_emb=self.user_emb,
            item_emb=self.item_emb
        )
        model.load_state_dict(torch.load(MODEL_DIR / "ranking_model.pt", map_location=DEVICE))
        model.eval()
        return model

    def recommend(self, user_id):

        user_vector = self.user_emb[user_id:user_id+1]
        faiss.normalize_L2(user_vector)

        scores, candidates = self.index.search(user_vector, TOP_K_RETRIEVAL)

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

        ranked_indices = torch.argsort(preds, descending=True)

        final_items = candidate_items[ranked_indices[:TOP_K_FEED]]

        return final_items.tolist()


if __name__ == "__main__":
    recommender = FeedRecommender()

    test_user = 42
    recs = recommender.recommend(test_user)

    print("Top 20 feed recommendations for user", test_user)
    print(recs)
