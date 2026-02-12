import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

RAW_DIR = Path("data/raw/ml-20m")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RATINGS_PATH = RAW_DIR / "ratings.csv"

MIN_INTERACTIONS_TO_SPLIT = 10  # users with fewer than this will have all interactions in train
IMPLICIT_RATING_THRESHOLD = 4.0  # rating >= this -> positive (1)

def load_ratings(path=RATINGS_PATH):
    print(f"Loading ratings from {path} ...")
    df = pd.read_csv(path)
    # ensure required cols
    assert set(["userId", "movieId", "rating", "timestamp"]).issubset(df.columns), "ratings.csv missing required columns"
    return df

def convert_to_implicit(df, threshold=IMPLICIT_RATING_THRESHOLD):
    print(f"Converting explicit ratings -> implicit with threshold >= {threshold}")
    df = df.copy()
    df["label"] = (df["rating"] >= threshold).astype(int)
    # keep only positive interactions (like 'engagements') - for recall/embedding training
    df_pos = df[df["label"] == 1].reset_index(drop=True)
    return df_pos

def time_aware_split(df_pos):
    print("Performing time-aware per-user split ...")
    # ensure timestamp is int
    df_pos["timestamp"] = df_pos["timestamp"].astype(int)
    train_parts, val_parts, test_parts = [], [], []

    # group by user and split
    for user_id, grp in tqdm(df_pos.groupby("userId"), desc="users"):
        grp_sorted = grp.sort_values("timestamp")
        n = len(grp_sorted)
        if n < MIN_INTERACTIONS_TO_SPLIT:
            train_parts.append(grp_sorted)
            continue
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        train_parts.append(grp_sorted.iloc[:train_end])
        val_parts.append(grp_sorted.iloc[train_end:val_end])
        test_parts.append(grp_sorted.iloc[val_end:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True) if len(val_parts) > 0 else pd.DataFrame(columns=train_df.columns)
    test_df = pd.concat(test_parts).reset_index(drop=True) if len(test_parts) > 0 else pd.DataFrame(columns=train_df.columns)

    return train_df, val_df, test_df

def build_id_mappings(train_df, val_df, test_df):
    # map all unique users/items appearing in any split to contiguous indices
    print("Building user/item id maps ...")
    all_users = pd.concat([train_df["userId"], val_df["userId"], test_df["userId"]]).unique()
    all_items = pd.concat([train_df["movieId"], val_df["movieId"], test_df["movieId"]]).unique()

    user_map = pd.DataFrame({"userId_orig": all_users})
    user_map = user_map.reset_index().rename(columns={"index": "user_idx"})
    item_map = pd.DataFrame({"movieId_orig": all_items})
    item_map = item_map.reset_index().rename(columns={"index": "item_idx"})

    # create fast lookup dicts
    user2idx = dict(zip(user_map["userId_orig"].values, user_map["user_idx"].values))
    item2idx = dict(zip(item_map["movieId_orig"].values, item_map["item_idx"].values))

    return user_map, item_map, user2idx, item2idx

def apply_mappings(df, user2idx, item2idx):
    df = df.copy()
    df["user_idx"] = df["userId"].map(user2idx).astype(np.int32)
    df["item_idx"] = df["movieId"].map(item2idx).astype(np.int32)
    # keep necessary columns
    return df[["userId", "movieId", "user_idx", "item_idx", "timestamp", "label"]]

def save_parquet(df, path: Path):
    print(f"Saving {path} (rows={len(df)}) ...")
    df.to_parquet(path, index=False)

def main():
    df = load_ratings()
    df_pos = convert_to_implicit(df)
    print(f"Positive interactions (rating >= {IMPLICIT_RATING_THRESHOLD}): {len(df_pos):,}")

    train_df, val_df, test_df = time_aware_split(df_pos)
    print("Split sizes: train={}, val={}, test={}".format(len(train_df), len(val_df), len(test_df)))

    user_map, item_map, user2idx, item2idx = build_id_mappings(train_df, val_df, test_df)

    # apply mappings
    train_mapped = apply_mappings(train_df, user2idx, item2idx)
    val_mapped = apply_mappings(val_df, user2idx, item2idx) if len(val_df) > 0 else val_df
    test_mapped = apply_mappings(test_df, user2idx, item2idx) if len(test_df) > 0 else test_df

    # Save
    save_parquet(train_mapped, OUT_DIR / "train.parquet")
    save_parquet(val_mapped, OUT_DIR / "val.parquet")
    save_parquet(test_mapped, OUT_DIR / "test.parquet")

    user_map.to_csv(OUT_DIR / "user_map.csv", index=False)
    item_map.to_csv(OUT_DIR / "item_map.csv", index=False)
    print(f"Saved user_map ({len(user_map)} users) and item_map ({len(item_map)} items)")

    # Print quick stats
    print("\n=== Quick stats ===")
    print("Train interactions:", len(train_mapped))
    print("Val interactions:", len(val_mapped))
    print("Test interactions:", len(test_mapped))
    print("Unique users:", len(user_map))
    print("Unique items:", len(item_map))
    # interactions per user distribution (train)
    if len(train_mapped) > 0:
        up = train_mapped.groupby("user_idx").size().describe()
        print("\nInteractions per user in train (desc):")
        print(up)

if __name__ == "__main__":
    main()
