import streamlit as st
import numpy as np
from utils.recommender import load_data, load_model, get_recommendations

# ===== Config =====
st.set_page_config(page_title="Movie Recommender", layout="wide")

# ===== State & Sidebar =====
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("🎬 Mood MovieBot")
if st.sidebar.button("🗑️ New chat"):
    st.session_state.history = []
    st.rerun()

st.header("🎥 Movie Mood Chatbot")
user_input = st.chat_input("What kind of movie are you in the mood for?")

# ===== Load static data =====
df = load_data()
model = load_model()
embeddings = np.load("data/embeddings.npy")

# ===== On user input =====
if user_input:
    if "forget" in user_input.lower():
        st.session_state.history = []
        st.chat_message("assistant").write("🧠 Memory cleared.")
    else:
        st.session_state.history.append(user_input)
        full_query = " ".join(st.session_state.history)
        recs = get_recommendations(full_query, df, embeddings, model)

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            st.subheader("🎯 Recommended Movies")
            cols = st.columns(len(recs))
            for i, movie in enumerate(recs):
                with cols[i]:
                    st.image(movie["poster"], use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    st.caption(movie["overview"][:180] + "…")

# ===== Footer =====
st.markdown("""
<hr style='margin-top:40px;'>
<div style='text-align:center;color:gray;font-size:0.9em'>Made with ❤️ by Suhaan Kareem · Powered by Streamlit & Transformers</div>
""", unsafe_allow_html=True)
