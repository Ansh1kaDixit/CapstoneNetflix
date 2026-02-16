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
if 'recs_shown' not in st.session_state:
    st.session_state.recs_shown = False
if 'last_recommendations' not in st.session_state:
    st.session_state.last_recommendations = []

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
                # NEW PREDICTION: Reset the seen titles list and recommendation state
                st.session_state.last_cluster = model.predict(vectorized_input)[0]
                st.session_state.seen_titles = []
                st.session_state.recs_shown = False
                st.session_state.last_recommendations = []
        else:
            st.warning("Please enter a description.")

# NOTE: the 'Suggest More' button is now a smart button rendered only when
# recommendations are visible (see logic below). The old always-visible
# button was removed.

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
    
    # SMART: show initial recommendations once, then let the "Suggest More"
    # button fetch additional unseen batches or reset when exhausted.
    def pick_and_store(n=5):
        available = full_cluster_df[~full_cluster_df['title'].isin(st.session_state.seen_titles)]
        if available.empty:
            return pd.DataFrame([])
        n_show = min(len(available), n)
        picks = available[['title', 'type', 'listed_in', 'release_year']].sample(n_show)
        st.session_state.seen_titles.extend(picks['title'].tolist())
        st.session_state.last_recommendations = picks.to_dict('records')
        st.session_state.recs_shown = True
        return picks

    # If we haven't yet shown the first batch after prediction, do so now.
    if not st.session_state.recs_shown:
        new_recommendations = pick_and_store(5)
    else:
        # Rehydrate the last recommendations for display
        if st.session_state.last_recommendations:
            new_recommendations = pd.DataFrame(st.session_state.last_recommendations)
        else:
            new_recommendations = pick_and_store(5)

    st.markdown(f"### 🍿 Recommendations (Unseen in this session)")
    if new_recommendations.empty:
        st.warning("No available recommendations for this cluster.")
    else:
        st.table(new_recommendations)
    st.caption(f"Currently tracking {len(st.session_state.seen_titles)} seen titles in this session.")

    # Smart Suggest button (only shown when recommendations are visible)
    remaining_df = full_cluster_df[~full_cluster_df['title'].isin(st.session_state.seen_titles)]
    if not remaining_df.empty:
        if st.button("🔄 Suggest More (Unseen Content)"):
            pick_and_store(5)
            st.experimental_rerun()
    else:
        # All titles have been seen in this session: offer reset-and-suggest
        if st.button("🔁 Reset & Suggest Again"):
            st.session_state.seen_titles = []
            pick_and_store(5)
            st.experimental_rerun()
else:
    st.info("Waiting for input... Paste a description and click 'Predict' to see results.")