import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import urllib.parse
import streamlit.components.v1 as components

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
    # precompute TF-IDF matrix for the corpus (used for semantic similarity)
    text_corpus = df['text_blob'].fillna('').astype(str).tolist()
    try:
        df_tfidf = vectorizer.transform(text_corpus)
    except Exception:
        # defensive: if transform fails, create an empty sparse matrix placeholder
        from scipy import sparse
        df_tfidf = sparse.csr_matrix((len(text_corpus), 0))
    return df, model, vectorizer, df_tfidf

df, model, vectorizer, df_tfidf = load_data()

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
    # Anchor the cluster glimpse so we can scroll to it programmatically
    st.markdown("<div id='cluster_glimpse'></div>", unsafe_allow_html=True)
    st.subheader(f"Glimpse into Cluster {selected_cluster}")
    # include inline Netflix/IMDb links and image column in the glimpse
    cluster_view = df[df['cluster_km'] == selected_cluster][[
        'title', 'type', 'listed_in', 'description', 'Netflix Link', 'IMDb Link', 'Image'
    ]].sample(10)
    st.dataframe(cluster_view[['title', 'type', 'listed_in', 'description']])
    # show inline action links (use CSV links, not a search query)
    for _, r in cluster_view.iterrows():
        nf_url = r.get('Netflix Link') or ''
        imdb_url = r.get('IMDb Link') or ''
        link_html = []
        if nf_url:
            link_html.append(f'<a href="{nf_url}" target="_blank" rel="noopener noreferrer">🔗 Open on Netflix — {r["title"]}</a>')
        if imdb_url:
            link_html.append(f'<a href="{imdb_url}" target="_blank" rel="noopener noreferrer">🎬 IMDb — {r["title"]}</a>')
        if link_html:
            st.markdown(" &nbsp;|&nbsp; ".join(link_html), unsafe_allow_html=True)
    # Scroll to the cluster glimpse element
    components.html(
        """
        <script>
          const el = document.getElementById('cluster_glimpse');
          if (el) { el.scrollIntoView({behavior: 'smooth', block: 'center'}); }
        </script>
        """,
        height=0,
    )
    st.session_state.scrolled_to_cluster_glimpse = True

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
if 'scrolled_to_recs' not in st.session_state:
    st.session_state.scrolled_to_recs = False
if 'scrolled_to_cluster_glimpse' not in st.session_state:
    st.session_state.scrolled_to_cluster_glimpse = False

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
                st.session_state.scrolled_to_recs = False
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
    def pick_and_store(n=5, source_df=None):
        # source_df: optional DataFrame to sample from (defaults to the full cluster)
        if source_df is None:
            source_df = full_cluster_df
        available = source_df[~source_df['title'].isin(st.session_state.seen_titles)]
        if available.empty:
            return pd.DataFrame([])
        n_show = min(len(available), n)
        # include Netflix/IMDb links and image so UI can render inline actions
        picks = available[[
            'title', 'type', 'listed_in', 'release_year', 'Netflix Link', 'IMDb Link', 'Image', 'Poster'
        ]].sample(n_show)
        st.session_state.seen_titles.extend(picks['title'].tolist())
        st.session_state.last_recommendations = picks.to_dict('records')
        st.session_state.recs_shown = True
        return picks

    # If we haven't yet shown the first batch after prediction, do so now.
    # --- stricter matching: use semantic similarity to avoid overly-broad results
    # compute cosine similarity between input and corpus TF-IDF rows
    SIM_THRESHOLD = 0.08  # tune this to make matches stricter (lower => broader)
    sim_series = None
    try:
        sim_scores = (vectorized_input @ df_tfidf.T).toarray()[0]
        sim_series = pd.Series(sim_scores, index=df.index)
    except Exception:
        sim_series = pd.Series([0.0] * len(df), index=df.index)

    # prefer high-similarity items inside the predicted cluster
    cluster_sim_scores = sim_series.loc[full_cluster_df.index]
    candidate_df = full_cluster_df.loc[cluster_sim_scores[cluster_sim_scores >= SIM_THRESHOLD].sort_values(ascending=False).index]

    if not st.session_state.recs_shown:
        if not candidate_df.empty:
            new_recommendations = pick_and_store(5, source_df=candidate_df)
        else:
            # no close semantic matches in predicted cluster — fall back but warn
            st.warning("Low semantic match for this description in the predicted cluster — results may be broad.")
            new_recommendations = pick_and_store(5)
    else:
        # Rehydrate the last recommendations for display
        if st.session_state.last_recommendations:
            new_recommendations = pd.DataFrame(st.session_state.last_recommendations)
        else:
            # when user clicks 'Suggest More' use the stricter candidate set if available
            if not candidate_df.empty:
                new_recommendations = pick_and_store(5, source_df=candidate_df)
            else:
                new_recommendations = pick_and_store(5)

    # Anchor recommendations for scrolling and display
    st.markdown("<div id='recommendations_anchor'></div>", unsafe_allow_html=True)
    st.markdown(f"### 🍿 Recommendations (Unseen in this session)")
    if new_recommendations.empty:
        st.warning("No available recommendations for this cluster.")
    else:
        # render cards with image + inline Netflix/IMDb links
        for rec in pd.DataFrame(new_recommendations).to_dict('records'):
            cols = st.columns([1, 4])
            with cols[0]:
                img = rec.get('Image') or rec.get('Poster')
                if img and isinstance(img, str) and img.strip():
                    st.image(img, width=120)
                else:
                    st.write('')
            with cols[1]:
                title = rec.get('title', 'Unknown')
                year = rec.get('release_year')
                header = f"**{title}** {f'({int(year)})' if year and not pd.isna(year) else ''}"
                st.markdown(header)
                if rec.get('listed_in'):
                    st.caption(rec.get('listed_in'))
                link_bits = []
                nf_link = rec.get('Netflix Link')
                imdb_link = rec.get('IMDb Link')
                if nf_link and isinstance(nf_link, str) and nf_link.strip():
                    link_bits.append(f'<a href="{nf_link}" target="_blank" rel="noopener noreferrer">🔗 View on Netflix</a>')
                if imdb_link and isinstance(imdb_link, str) and imdb_link.strip():
                    link_bits.append(f'<a href="{imdb_link}" target="_blank" rel="noopener noreferrer">🎬 IMDb</a>')
                if link_bits:
                    st.markdown(' &nbsp;|&nbsp; '.join(link_bits), unsafe_allow_html=True)
        # Auto-scroll to recommendations on first display
        if st.session_state.recs_shown and not st.session_state.scrolled_to_recs:
            components.html(
                """
                <script>
                  const el = document.getElementById('recommendations_anchor');
                  if (el) { el.scrollIntoView({behavior: 'smooth', block: 'center'}); }
                </script>
                """,
                height=0,
            )
            st.session_state.scrolled_to_recs = True
    st.caption(f"Currently tracking {len(st.session_state.seen_titles)} seen titles in this session.")

    # Smart Suggest button (only shown when recommendations are visible)
    remaining_df = full_cluster_df[~full_cluster_df['title'].isin(st.session_state.seen_titles)]
    if not remaining_df.empty:
        if st.button("🔄 Suggest More (Unseen Content)"):
            pick_and_store(5)
    else:
        # All titles have been seen in this session: offer reset-and-suggest
        if st.button("🔁 Reset & Suggest Again"):
            st.session_state.seen_titles = []
            pick_and_store(5)
else:
    st.info("Waiting for input... Paste a description and click 'Predict' to see results.")