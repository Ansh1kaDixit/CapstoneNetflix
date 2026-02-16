import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- 1. Setup & Downloads ---
@st.cache_resource
def download_nltk_data():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

download_nltk_data()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- 2. Load Artifacts ---
@st.cache_resource
def load_data():
    df = pd.read_csv('netflix_final_clustered_data.csv')
    model = pickle.load(open('netflix_kmeans_model.pkl', 'rb'))
    vectorizer = pickle.load(open('netflix_tfidf_vectorizer.pkl', 'rb'))
    return df, model, vectorizer

df, model, vectorizer = load_data()

# --- 3. Cleaning Function ---
def advanced_clean(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned_words)

# --- 4. Streamlit Layout ---
st.set_page_config(page_title="Netflix Theme Predictor", page_icon="🍿")
st.title("🎬 Netflix Content Strategy & Recommendation Engine")

# Testing Samples in Sidebar
st.sidebar.header("🧪 Test Sample Descriptions")
st.sidebar.info("Copy and paste these into the box to see the model work accurately:")

samples = {
    "Kids/Animation": "An animated musical adventure for young children featuring talking animals and catchy songs about friendship.",
    "Documentary": "A deep-dive investigative documentary exploring real-life crime scenes and forensic evidence.",
    "Horror/Thriller": "A dark psychological thriller where a group of teenagers discovers a haunted house in the middle of a forest.",
    "International Drama": "A sweeping romantic period drama set in 19th-century Europe, exploring themes of forbidden love and betrayal."
}

for label, text in samples.items():
    st.sidebar.text_area(f"Sample for {label}:", text, height=100)

# Main Prediction Area
st.subheader("Analyze Content Description")
user_input = st.text_area("Paste a detailed description here:", height=150, help="Short titles like 'Baby' don't provide enough data. Try 2-3 sentences.")

if st.button("Predict Cluster & Recommend"):
    if user_input.strip():
        cleaned_text = advanced_clean(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        
        # Check if the model recognizes any words
        if vectorized_input.nnz == 0:
            st.error("⚠️ **The model doesn't recognize those keywords.**")
            st.warning("Short phrases like 'baby theme songs' are too vague. Please add more descriptive details about the genre, plot, or characters.")
        else:
            cluster_id = model.predict(vectorized_input)[0]
            st.success(f"Predicted Strategic Cluster ID: **{cluster_id}**")
            
            # Recommendation Logic
            st.markdown(f"### 🍿 Similar Titles in Cluster {cluster_id}")
            cluster_df = df[df['cluster_km'] == cluster_id]
            n_samples = min(len(cluster_df), 5)
            recommendations = cluster_df[['title', 'type', 'listed_in', 'release_year']].sample(n_samples)
            st.table(recommendations)
    else:
        st.warning("Please enter some