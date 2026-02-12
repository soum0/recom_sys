import os

# Mac runtime safety
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from fastapi import FastAPI, HTTPException
import torch
import numpy as np
import faiss
from pathlib import Path

MODEL_DIR = Path("models")

TOP_K_RETRIEVAL = 100
TOP_K_FEED = 20
DEVICE = "cpu"

app = FastAPI(title="Instagram-Style Recommendation API")


# -------------------------------
# Load System Once (On Startup)
# -------------------------------

class RecommenderService:

    def __init__(self):
        print("Loading recommendation system...")

        self.user_emb = np.load(MODEL_DIR / "user_embeddings.npy")
        self.item_emb = np.load(MODEL_DIR / "item_embeddings.npy")

        self.index = faiss.read_index(str(MODEL_DIR / "faiss_item.index"))

        self.ranking_model = self._load_ranker()

        faiss.normalize_L2(self.item_emb)

        print("Recommendation system ready.")

    def _load_ranker(self):
        from src.train_ranker import RankingModel

        model = RankingModel(
            user_emb=self.user_emb,
            item_emb=self.item_emb
        )

        model.load_state_dict(
            torch.load(MODEL_DIR / "ranking_model.pt", map_location=DEVICE)
        )

        model.eval()
        return model

    def recommend(self, user_id: int):

        if user_id >= len(self.user_emb):
            raise ValueError("Invalid user_id")

        user_vec = self.user_emb[user_id:user_id+1]
        faiss.normalize_L2(user_vec)

        scores, candidates = self.index.search(user_vec, TOP_K_RETRIEVAL)

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

        final_items = candidate_items[ranked_idx[:TOP_K_FEED]]

        return final_items.tolist()


# Initialize system (singleton)
recommender = RecommenderService()


# -------------------------------
# API Endpoints
# -------------------------------

@app.get("/")
def home():
    return {"status": "Recommendation API running"}


@app.get("/recommend")
def recommend(user_id: int):

    try:
        results = recommender.recommend(user_id)
        return {
            "user_id": user_id,
            "recommendations": results
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
