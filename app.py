import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import urllib.parse
import base64
import html
import streamlit.components.v1 as components

# Fallback poster used when a title has no image/poster in the CSV.
# Use a poster-shaped SVG (2:3) with a large red "N" so the placeholder
# matches poster aspect ratio and loads at the same visual size as real posters.
_FALLBACK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 630" preserveAspectRatio="xMidYMid meet">'
    '<rect x="0" y="0" width="420" height="630" rx="14" fill="#141414"/>'
    # centered group with a large red 'N' — use alignment-baseline and text-anchor
    '<g transform="translate(210,315)">'
    '<text x="0" y="0" font-family="Arial, Helvetica, sans-serif" font-size="260" '
    'fill="#E50914" font-weight="800" text-anchor="middle" alignment-baseline="central">N</text>'
    '</g>'
    '</svg>'
)
# Use base64-encoded data URI for maximum cross-browser reliability
FALLBACK_POSTER = 'data:image/svg+xml;base64,' + base64.b64encode(_FALLBACK_SVG.encode('utf-8')).decode('ascii')

# Default similarity threshold (used by slider reset)
DEFAULT_SIM_THRESHOLD = 0.08

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


def excerpt(text: str, max_len: int = 160) -> str:
    """Return a short, single-line excerpt from a longer description."""
    if not text:
        return ""
    s = str(text).strip()
    if len(s) <= max_len:
        return s
    cut = s[:max_len].rsplit(' ', 1)[0]
    return cut + '...'


def render_image_with_fallback(img_url, width=120, alt='poster'):
    """Render an <img> with an onerror handler that replaces a failing src with
    the poster-shaped `FALLBACK_POSTER` (keeps a consistent 2:3 aspect ratio).
    Use this instead of `st.image` so broken/missing URLs gracefully fall back.
    """
    # normalize/validate incoming URL
    if not img_url or (isinstance(img_url, float) and pd.isna(img_url)) or not str(img_url).strip():
        img_url = FALLBACK_POSTER
    height = int(width * 1.5)
    # embed fallback as a quoted JS string (FALLBACK_POSTER is URL-encoded SVG)
    html = (
        f"<img src=\"{img_url}\" alt=\"{alt}\" width=\"{width}\" height=\"{height}\" "
        f"style=\"object-fit:cover;border-radius:6px;border:1px solid #ddd;\" "
        f"onerror=\"this.onerror=null;this.src='{FALLBACK_POSTER}';\"/>")
    st.markdown(html, unsafe_allow_html=True)


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
    # render a compact list with poster + title + inline links (use CSV image if present)
    for r in cluster_view.to_dict('records'):
        cols = st.columns([1, 4])
        with cols[0]:
            img = r.get('Image') or r.get('Poster')
            render_image_with_fallback(img, width=100, alt=r.get('title', 'poster'))
        with cols[1]:
            st.markdown(f"**{r.get('title','Unknown')}**")
            if r.get('listed_in'):
                st.caption(r.get('listed_in'))
            # short description excerpt
            desc = r.get('description') or r.get('Summary') or ''
            if desc:
                st.markdown(f"<div style='color:#555;font-size:13px'>{excerpt(desc, 160)}</div>", unsafe_allow_html=True)
            links = []
            if r.get('Netflix Link'):
                links.append(f'<a href="{r.get("Netflix Link")}" target="_blank" rel="noopener noreferrer">🔗 Netflix</a>')
            if r.get('IMDb Link'):
                links.append(f'<a href="{r.get("IMDb Link")}" target="_blank" rel="noopener noreferrer">🎬 IMDb</a>')
            if links:
                st.markdown(' &nbsp;|&nbsp; '.join(links), unsafe_allow_html=True)
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
if 'last_query' not in st.session_state:
    st.session_state.last_query = ''
if 'sim_threshold' not in st.session_state:
    st.session_state.sim_threshold = DEFAULT_SIM_THRESHOLD
if 'sim_threshold_prev' not in st.session_state:
    st.session_state.sim_threshold_prev = DEFAULT_SIM_THRESHOLD

user_input = st.text_area(
    "Paste Content Description Here:", 
    height=150, 
    placeholder="e.g., A group of survivors must navigate a post-apocalyptic world..."
)

# Similarity slider with Reset / Undo controls
col_s, col_reset, col_undo = st.columns([6, 1, 1])
with col_s:
    st.slider(
        "Similarity threshold (higher = stricter)",
        min_value=0.0,
        max_value=0.50,
        step=0.01,
        key='sim_threshold',
        help="Higher value => fewer but more semantically-similar recommendations",
    )
with col_reset:
    if st.button("Reset to default"):
        st.session_state.sim_threshold = DEFAULT_SIM_THRESHOLD
with col_undo:
    if st.button("Undo"):
        prev = st.session_state.get('sim_threshold_prev')
        if prev is not None:
            st.session_state.sim_threshold, st.session_state.sim_threshold_prev = prev, st.session_state.sim_threshold
st.caption("Tip: move the slider to adjust how closely recommendations must match your description.")

# local alias used elsewhere in the module
sim_threshold = st.session_state.sim_threshold

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Predict & Recommend"):
        if user_input.strip():
            # capture the slider value used for this prediction so the user can
            # undo back to it if needed
            st.session_state.sim_threshold_prev = st.session_state.get('sim_threshold', DEFAULT_SIM_THRESHOLD)
            cleaned_text = advanced_clean(user_input)
            # persist cleaned input so subsequent UI interactions (slider)
            # can recompute similarity without re-clicking Predict
            st.session_state.last_query = cleaned_text
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
        # include description + Netflix/IMDb links and image so UI can render inline actions
        picks = available[[
            'title', 'type', 'listed_in', 'description', 'release_year',
            'Netflix Link', 'IMDb Link', 'Image', 'Poster'
        ]].sample(n_show)
        st.session_state.seen_titles.extend(picks['title'].tolist())
        st.session_state.last_recommendations = picks.to_dict('records')
        st.session_state.recs_shown = True
        return picks

    # If we haven't yet shown the first batch after prediction, do so now.
    # --- stricter matching: use semantic similarity to avoid overly-broad results
    # compute cosine similarity between the last query (persisted) and corpus
    SIM_THRESHOLD = sim_threshold  # user-controlled via slider
    sim_series = None
    if st.session_state.get('last_query'):
        try:
            v_input = vectorizer.transform([st.session_state.last_query])
            sim_scores = (v_input @ df_tfidf.T).toarray()[0]
            sim_series = pd.Series(sim_scores, index=df.index)
        except Exception:
            sim_series = pd.Series([0.0] * len(df), index=df.index)
    else:
        # no persisted query available (no prediction yet)
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
                render_image_with_fallback(img, width=120, alt=rec.get('title', 'poster'))
            with cols[1]:
                title = rec.get('title', 'Unknown')
                year = rec.get('release_year')
                header = f"**{title}** {f'({int(year)})' if year and not pd.isna(year) else ''}"
                st.markdown(header)
                if rec.get('listed_in'):
                    st.caption(rec.get('listed_in'))
                # short description excerpt for recommendation cards
                desc = rec.get('description') or rec.get('Summary') or ''
                if desc:
                    st.markdown(
                        f"<div style='color:#555;font-size:13px'>{excerpt(desc, 160)}</div>",
                        unsafe_allow_html=True,
                    )
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