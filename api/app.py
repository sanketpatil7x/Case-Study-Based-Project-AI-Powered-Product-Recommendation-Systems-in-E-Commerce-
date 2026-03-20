"""
AI-Powered Recommendation System

This module provides a Streamlit-based interface for generating
personalized recommendations using a Matrix Factorization model.
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ----------------------------
# Load Data
# ----------------------------

@st.cache_data
def load_data():
    R_pred = np.load("data/processed/R_pred.npy")
    train_df = pd.read_csv("data/processed/train.csv")

    with open("data/processed/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)

    with open("data/processed/item_encoder.pkl", "rb") as f:
        item_encoder = pickle.load(f)

    return R_pred, train_df, user_encoder, item_encoder


@st.cache_data
def load_movies():
    movies = pd.read_csv(
        "data/raw/u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "title"]
    )
    return movies


R_pred, train_df, user_encoder, item_encoder = load_data()
movies_df = load_movies()

# ----------------------------
# Recommendation Function
# ----------------------------

def recommend(user_id, k=10):
    """
    Generate top-k recommendations for a given user.

    Returns:
        (items, scores) or None
    """
    try:
        user_enc = user_encoder.transform([user_id])[0]
    except Exception:
        return None

    scores = R_pred[user_enc].copy()

    # Remove already seen items
    seen_items = train_df[train_df['user_id_enc'] == user_enc]['item_id_enc'].values
    scores[seen_items] = -np.inf

    # Top-K
    top_items_enc = np.argsort(scores)[::-1][:k]
    top_scores = scores[top_items_enc]

    # Decode
    top_items = item_encoder.inverse_transform(top_items_enc)

    return top_items, top_scores


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="AI Recommender", layout="centered")

st.title("AI-Powered Recommendation System")
st.markdown("Enter a **User ID** to get personalized movie recommendations.")

user_input = st.number_input("User ID", min_value=1, step=1)

if st.button("Get Recommendations"):
    result = recommend(user_input, k=10)

    if result is None:
        st.error("❌ User not found. Try a valid ID (1–943).")
    else:
        items, scores = result

        st.success("✅ Top Recommendations:")

        for i, (item, score) in enumerate(zip(items, scores), 1):

            # Get movie name
            movie = movies_df[movies_df['item_id'] == item]['title'].values

            title = movie[0] if len(movie) > 0 else f"Item {item}"

            st.markdown(
                f"""
                **{i}. 🎬 {title}**  
                   Score: `{score:.3f}`
                """
            )