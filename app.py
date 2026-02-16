import streamlit as st
import pandas as pd
import pickle
import re

# Set page configuration
st.set_page_config(page_title="Netflix Content Clustering", layout="wide")

# --- 1. Load Data and Models ---
@st.cache_resource
def load_artifacts():
    # Load the processed data
    df = pd.read_csv('netflix_final_clustered_data.csv')
    # Load the K-Means model
    model = pickle.load(open('netflix_kmeans_model.pkl', 'rb'))
    # Load the TF-IDF Vectorizer
    vectorizer = pickle.load(open('netflix_tfidf_vectorizer.pkl', 'rb'))
    return df, model, vectorizer

df, model, vectorizer = load_artifacts()

# --- 2. Helper Functions ---
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    return text

# --- 3. Streamlit UI Layout ---
st.title("🎬 Netflix Content Recommendation & Clustering")
st.markdown("""
This app uses a **K-Means Clustering model** to categorize movie/TV show descriptions into thematic clusters. 
Paste a description below to see its predicted category!
""")

# User Input Section
st.subheader("Analyze New Content")
user_input = st.text_area("Enter Movie/TV Show Description:", placeholder="e.g., A group of friends embark on a journey to find a hidden treasure...")

if st.button("Predict Cluster"):
    if user_input:
        # Step A: Pre-process
        cleaned = clean_text(user_input)
        
        # Step B: Vectorize
        vectorized = vectorizer.transform([cleaned])
        
        # Step C: Predict
        cluster_id = model.predict(vectorized)[0]
        
        # Display Result
        st.success(f"Predicted Strategic Cluster ID: **{cluster_id}**")
        
        # Show Cluster Themes (Optional: Add your Word Cloud logic here)
        st.info("Searching for similar titles in this cluster...")
        
        # Step D: Show similar movies from the same cluster
        similar_content = df[df['cluster_km'] == cluster_id][['title', 'type', 'listed_in']].head(5)
        st.table(similar_content)
    else:
        st.warning("Please enter a description first.")

# --- 4. Sidebar Exploration ---
st.sidebar.header("Explore Clusters")
cluster_choice = st.sidebar.selectbox("Select a Cluster to explore titles:", range(6))
if st.sidebar.button("Show Titles"):
    st.subheader(f"Titles in Cluster {cluster_choice}")
    st.dataframe(df[df['cluster_km'] == cluster_choice][['title', 'type', 'listed_in', 'release_year']].sample(10))