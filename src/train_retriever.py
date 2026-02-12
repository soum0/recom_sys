import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

DEVICE = "cpu"
EMBED_DIM = 64
BATCH_SIZE = 2048
EPOCHS = 3
LR = 0.001

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.parquet"

class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = df["user_idx"].values
        self.items = df["item_idx"].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

class RetrieverModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

    def forward(self, users, items):
        u = self.user_embed(users)
        i = self.item_embed(items)
        scores = (u * i).sum(dim=1)
        return scores

def main():
    print("Loading training data...")
    df = pd.read_parquet(TRAIN_PATH)

    num_users = df["user_idx"].nunique()
    num_items = df["item_idx"].nunique()

    print("Users:", num_users)
    print("Items:", num_items)

    dataset = InteractionDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = RetrieverModel(num_users, num_items, EMBED_DIM).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for users, items, labels in tqdm(loader):
            users = users.to(DEVICE)
            items = items.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save embeddings
    user_emb = model.user_embed.weight.detach().cpu().numpy()
    item_emb = model.item_embed.weight.detach().cpu().numpy()

    np.save(MODEL_DIR / "user_embeddings.npy", user_emb)
    np.save(MODEL_DIR / "item_embeddings.npy", item_emb)

    print("Embeddings saved successfully.")

if __name__ == "__main__":
    main()
