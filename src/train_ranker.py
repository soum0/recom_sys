import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

DEVICE = "cpu"
BATCH_SIZE = 1024
EPOCHS = 3
LR = 0.001
EMBED_DIM = 64

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

RANK_PATH = DATA_DIR / "ranking_train.parquet"

class RankingDataset(Dataset):
    def __init__(self, df):
        self.users = df["user_idx"].values
        self.items = df["item_idx"].values
        self.sim = df["sim_score"].values
        self.popularity = df["popularity"].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.sim[idx], dtype=torch.float32),
            torch.tensor(self.popularity[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

class RankingModel(nn.Module):
    def __init__(self, user_emb, item_emb):
        super().__init__()

        self.user_embedding = nn.Embedding.from_pretrained(
            torch.tensor(user_emb), freeze=True
        )
        self.item_embedding = nn.Embedding.from_pretrained(
            torch.tensor(item_emb), freeze=True
        )

        input_dim = EMBED_DIM * 2 + 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_ids, item_ids, sim_score, popularity):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)

        x = torch.cat([
            u,
            i,
            sim_score.unsqueeze(1),
            popularity.unsqueeze(1)
        ], dim=1)

        out = self.mlp(x)
        return out.squeeze()


def main():
    print("Loading ranking dataset...")
    df = pd.read_parquet(RANK_PATH)

    print("Loading embeddings...")
    user_emb = np.load(MODEL_DIR / "user_embeddings.npy")
    item_emb = np.load(MODEL_DIR / "item_embeddings.npy")

    dataset = RankingDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = RankingModel(user_emb, item_emb).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting ranking model training...")

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for users, items, sim, pop, labels in tqdm(loader):
            users = users.to(DEVICE)
            items = items.to(DEVICE)
            sim = sim.to(DEVICE)
            pop = pop.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(users, items, sim, pop)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "ranking_model.pt")
    print("Ranking model saved successfully.")


if __name__ == "__main__":
    main()
