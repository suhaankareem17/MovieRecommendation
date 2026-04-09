import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# ====== Load data ======
df = pd.read_csv("data/movies_metadata.csv", low_memory=False)
df = df[['title', 'overview']]
df = df.dropna(subset=['overview']).reset_index(drop=True)

# ====== Encode & save ======
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['overview'].tolist(), show_progress_bar=True)

os.makedirs("data", exist_ok=True)
np.save("data/embeddings.npy", embeddings)
print("✅ Embeddings saved to data/embeddings.npy")
