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
    # Ensure these filenames match your saved files exactly
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

# --- 4. Sidebar: Branding & Tools ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.title("Navigation")
st.sidebar.info("Developed by **Anshika Dixit** | Unsupervised ML Project")

st.sidebar.markdown("---")

# Test Samples Dropdown in Sidebar
st.sidebar.header("🧪 Test Samples")
sample_options = {
    "Select a sample...": "",
    "Kids/Animation": "An animated musical adventure for young children featuring talking animals and catchy songs about friendship and learning.",
    "Documentary": "A deep-dive investigative documentary exploring real-life crime scenes, forensic evidence, and interviews with experts.",
    "Horror/Thriller": "A dark psychological thriller where a group of teenagers discovers a haunted house in the middle of a forest and must survive the night.",
    "International Drama": "A sweeping romantic period drama set in 19th-century Europe, exploring themes of forbidden love, social class, and betrayal."
}

selected_sample_label = st.sidebar.selectbox("Choose a sample to copy:", list(sample_options.keys()))
if selected_sample_label != "Select a sample...":
    st.sidebar.code(sample_options[selected_sample_label], language=None)
    st.sidebar.caption("Copy the text above and paste it into the main box.")

st.sidebar.markdown("---")

# Cluster Explorer in Sidebar
st.sidebar.header("🔍 Cluster Explorer")
selected_cluster = st.sidebar.slider("Select Cluster ID to view titles", 0, 5, 0)
if st.sidebar.button("Show Random Titles"):
    st.subheader(f"Glimpse into Cluster {selected_cluster}")
    cluster_view = df[df['cluster_km'] == selected_cluster][['title', 'type', 'listed_in', 'description']].sample(10)
    st.dataframe(cluster_view)

# --- 5. Main Prediction Area ---
st.title("🎬 Netflix Content Strategy & Recommendation Engine")
st.markdown("""
This application uses a **K-Means Clustering model** to categorize Netflix titles based on their thematic DNA. 
Paste a detailed description below to identify its strategic cluster.
""")

user_input = st.text_area(
    "Paste Content Description Here:", 
    height=200, 
    placeholder="e.g., A high-octane science fiction journey into outer space where a team of soldiers fights an alien invasion..."
)

if st.button("Predict & Recommend"):
    if user_input.strip():
        cleaned_text = advanced_clean(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        
        # Zero-Vector Check
        if vectorized_input.nnz == 0:
            st.error("⚠️ **The model doesn't recognize those keywords.**")
            st.warning("Input is too vague. Please add more descriptive details about the genre, plot, or characters (use the samples in the sidebar for reference).")
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
        st.warning("Please enter some text to analyze.")