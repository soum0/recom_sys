import streamlit as st
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw/ml-20m")

# Load item index to movieId mapping
item_map = pd.read_csv(DATA_DIR / "item_map.csv")

# Load movie metadata
movies_df = pd.read_csv(RAW_DIR / "movies.csv")

# Create lookup dictionaries
idx_to_movieid = dict(zip(item_map["item_idx"], item_map["movieId_orig"]))
movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))


API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="Instagram-Style Recommender", layout="centered")

st.title("📸 Instagram Explore-Style Recommendation System")

st.markdown("""
This demo shows a **two-stage recommender system**:
- Neural embedding retrieval (FAISS)
- Deep ranking model  
""")

# User input
user_id = st.number_input(
    "Enter User ID",
    min_value=0,
    max_value=138286,
    value=42,
    step=1
)

if st.button("🚀 Get Recommendations"):
    with st.spinner("Fetching personalized feed..."):
        response = requests.get(API_URL, params={"user_id": user_id})

        if response.status_code == 200:
            data = response.json()
            recs = data["recommendations"]

            st.success("Personalized Feed Generated!")

            st.subheader("Top Recommended Items")
            for rank, item_idx in enumerate(recs, start=1):

                movie_id = idx_to_movieid.get(item_idx, None)
                title = movieid_to_title.get(movie_id, "Unknown Movie")

                st.write(f"#{rank} ➜ 🎬 {title}")


        else:
            st.error("API Error. Is FastAPI server running?")
