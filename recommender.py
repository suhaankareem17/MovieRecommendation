import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    df = pd.read_csv("data/movies_metadata.csv", low_memory=False)
    df = df[['title', 'overview', 'poster_path', 'release_date', 'popularity']]
    df = df.dropna(subset=['overview'])
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    return df.reset_index(drop=True)

from sentence_transformers import SentenceTransformer

def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


def fetch_poster_url(poster_path):
    if pd.isna(poster_path):
        return "https://via.placeholder.com/300x450?text=No+Image"
    return f"https://image.tmdb.org/t/p/w500{poster_path}"

def get_recommendations(query, df, embeddings, model, top_n=5, prefer_popular=True):
    query_emb = model.encode([query])[0]
    sims = cosine_similarity([query_emb], embeddings)[0]

    df_local = df.copy()
    df_local['score'] = sims
    df_local['norm_pop'] = (df_local['popularity'] - df_local['popularity'].min()) / (df_local['popularity'].max() - df_local['popularity'].min() + 1e-6)
    df_local['norm_year'] = (df_local['year'] - df_local['year'].min()) / (df_local['year'].max() - df_local['year'].min() + 1e-6)

    pop_w, year_w = (0.4, 0.4) if prefer_popular else (0.1, 0.1)
    if any(k in query.lower() for k in ["old", "classic"]):
        pop_w = 0.05
    if any(k in query.lower() for k in ["obscure", "less known"]):
        pop_w = -0.3

    df_local['final'] = df_local['score'] + pop_w * df_local['norm_pop'] + year_w * df_local['norm_year']
    top = df_local.sort_values("final", ascending=False).head(top_n)

    return [{
        "title": r['title'],
        "overview": r['overview'],
        "poster": fetch_poster_url(r['poster_path'])
    } for _, r in top.iterrows()]
