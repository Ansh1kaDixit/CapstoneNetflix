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

# --- 4. Sidebar: Branding & Test Samples ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.title("Navigation")
st.sidebar.info("Developed by **Anshika Dixit** | Unsupervised ML Project")

st.sidebar.markdown("---")
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

st.sidebar.markdown("---")
st.sidebar.header("🔍 Dataset Explorer")
selected_cluster = st.sidebar.slider("Select Cluster ID to view", 0, 5, 0)
if st.sidebar.button("Show Random Titles from Dataset"):
    st.subheader(f"Glimpse into Cluster {selected_cluster}")
    cluster_view = df[df['cluster_km'] == selected_cluster][['title', 'type', 'listed_in', 'description']].sample(10)
    st.dataframe(cluster_view)

# --- 5. Main Prediction Area ---
st.title("🎬 Netflix Content Strategy & Recommendation Engine")

# Initialize Session States
if 'last_cluster' not in st.session_state:
    st.session_state.last_cluster = None
if 'seen_titles' not in st.session_state:
    st.session_state.seen_titles = []

user_input = st.text_area(
    "Paste Content Description Here:", 
    height=150, 
    placeholder="e.g., A group of survivors must navigate a post-apocalyptic world..."
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Predict & Recommend"):
        if user_input.strip():
            cleaned_text = advanced_clean(user_input)
            vectorized_input = vectorizer.transform([cleaned_text])
            
            if vectorized_input.nnz == 0:
                st.error("⚠️ The model doesn't recognize those keywords.")
                st.session_state.last_cluster = None
            else:
                # NEW PREDICTION: Reset the seen titles list
                st.session_state.last_cluster = model.predict(vectorized_input)[0]
                st.session_state.seen_titles = [] 
        else:
            st.warning("Please enter a description.")

with col2:
    # "Suggest More" Button logic
    suggest_more = st.button("🔄 Suggest More (Unseen Content)")

# Logic for displaying recommendations
if st.session_state.last_cluster is not None:
    st.success(f"Strategic Cluster ID: **{st.session_state.last_cluster}**")
    
    # 1. Filter by cluster
    full_cluster_df = df[df['cluster_km'] == st.session_state.last_cluster]
    
    # 2. FILTER OUT TITLES ALREADY SEEN
    available_df = full_cluster_df[~full_cluster_df['title'].isin(st.session_state.seen_titles)]
    
    if available_df.empty:
        st.warning("You have viewed all titles in this cluster! Resetting history...")
        st.session_state.seen_titles = []
        available_df = full_cluster_df
    
    # 3. Pick 5 new titles
    n_to_show = min(len(available_df), 5)
    new_recommendations = available_df[['title', 'type', 'listed_in', 'release_year']].sample(n_to_show)
    
    # 4. Add these new titles to the "seen" list so they don't show up next time
    st.session_state.seen_titles.extend(new_recommendations['title'].tolist())
    
    st.markdown(f"### 🍿 Recommendations (Unseen in this session)")
    st.table(new_recommendations)
    st.caption(f"Currently tracking {len(st.session_state.seen_titles)} seen titles in this session.")
else:
    st.info("Waiting for input... Paste a description and click 'Predict' to see results.")