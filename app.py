import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- 1. Initial Setup & NLTK Downloads ---
# These are required for the text cleaning to work on Streamlit Cloud
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
    # Make sure these filenames match exactly what you saved in your notebook
    df = pd.read_csv('netflix_final_clustered_data.csv')
    model = pickle.load(open('netflix_kmeans_model.pkl', 'rb'))
    vectorizer = pickle.load(open('netflix_tfidf_vectorizer.pkl', 'rb'))
    return df, model, vectorizer

try:
    df, model, vectorizer = load_data()
except FileNotFoundError:
    st.error("Required files (.pkl or .csv) not found. Please ensure they are in the same directory as app.py")

# --- 3. Text Pre-processing Function ---
def advanced_clean(text):
    # Match the logic used in netflix_ml.ipynb
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    words = text.split()
    # Lemmatize and remove stopwords
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned_words)

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Netflix Content Clustering", page_icon="🎬", layout="wide")

st.title("🎬 Netflix Content Strategy & Recommendation Engine")
st.markdown("""
This application uses an **Unsupervised Machine Learning (K-Means)** model to categorize Netflix titles based on their 'thematic DNA'.
Paste a movie or TV show description below to find its strategic cluster and similar titles.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Analyze New Content")
    user_input = st.text_area(
        "Enter Description/Keywords:", 
        height=200, 
        placeholder="e.g., A group of survivors must navigate a post-apocalyptic world filled with zombies..."
    )
    
    predict_button = st.button("Predict Cluster & Recommend")

with col2:
    st.subheader("Results")
    if predict_button:
        if user_input.strip():
            # Process and Predict
            cleaned_text = advanced_clean(user_input)
            
            # Check if input is too short after cleaning
            if not cleaned_text:
                st.error("Input is too vague. Please provide more descriptive keywords.")
            else:
                vectorized_input = vectorizer.transform([cleaned_text])
                cluster_id = model.predict(vectorized_input)[0]
                
                st.success(f"Predicted Strategic Cluster ID: **{cluster_id}**")
                
                # Show related titles from the same cluster
                st.markdown(f"### 🍿 Similar Titles in Cluster {cluster_id}")
                # We use .sample() so the list changes and feels fresh
                recommendations = df[df['cluster_km'] == cluster_id][['title', 'type', 'listed_in', 'release_year']].sample(5)
                st.table(recommendations)
        else:
            st.warning("Please enter a description to get a recommendation.")

# --- 5. Sidebar Explorer ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.markdown("---")
st.sidebar.header("Cluster Explorer")
selected_cluster = st.sidebar.slider("Select Cluster ID", 0, 5, 0)

if st.sidebar.button("Explore Titles"):
    st.subheader(f"Glimpse into Cluster {selected_cluster}")
    cluster_view = df[df['cluster_km'] == selected_cluster][['title', 'type', 'listed_in', 'description']].sample(10)
    st.dataframe(cluster_view)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Anshika Dixit | Unsupervised ML Project")