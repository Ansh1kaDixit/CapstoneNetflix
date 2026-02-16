This README provides a professional overview of the project, focusing on the strategic findings and technical implementation. You can use it as the main landing page for your GitHub repository.

🎬 Netflix Content Strategy & Clustering Analysis
Live demo: https://netflixcapstone.streamlit.app/
📌 Project Overview
This project involves a comprehensive unsupervised machine learning analysis of the Netflix catalog. By merging standard metadata with external IMDb ratings, this study moves beyond simple categorization to provide a quality-centric view of Netflix's evolution. The core goal is to understand the "thematic DNA" of content and automate the discovery of "Micro-genres" to enhance user recommendation systems.

Project Type: Unsupervised ML (Clustering)

Author: Anshika Dixit

Contribution: Individual

🔍 The Problem Statement & Business Context
Since 2010, Netflix’s library has seen a dramatic shift. While the number of movies has stabilized, the volume of TV shows has tripled.
Objectives:

Analyze the Pivot: Investigate the shift toward episodic content and its impact on user engagement.

Identify Quality Hubs: Pinpoint geographical regions producing the highest-rated content.

Automate Discovery: Use NLP to cluster content into 6 strategic pillars to power hyper-personalized user "rows."

💡 Key Strategic Findings
1. The Quality-Volume Paradox
Finding: While the US leads in total production volume, countries like India and South Korea serve as "Quality Hubs," producing content that achieves significantly higher average IMDb scores.

Recommendation: Netflix should continue aggressive investment in localized "Originals" from these hubs, as they offer a superior ROI for prestige content globally.

2. TV Show Dominance
Finding: TV shows generally maintain higher critical engagement than movies on the platform. This validates the business shift toward episodic content, which drives long-term subscriber retention ("stickiness").

3. Thematic Content Pillars
Through K-Means Clustering, we successfully identified 6 core thematic segments:

International Romantic Dramas

Kids & Family Animation

Documentary & True-Crime

Action, Sci-Fi & Adventure

Stand-Up Comedy

Scripted Regional Content (focused on India/Asia)

🛠️ Technical Workflow
1. NLP Pipeline
Pre-processing: Regex cleaning, lowercasing, and Lemmatization to normalize text.

Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency) with 5,000 features and n-gram ranges (1, 2) to capture context.

2. Clustering Algorithms
We evaluated three models to ensure robust grouping:

K-Means (Final Model): Achieved the best separation with a Silhouette Score of 0.0162.

Hierarchical Clustering: Used Dendrogram analysis to visualize the "genealogy" of sub-genres.

DBSCAN: Used to identify "Outliers"—highly unique, non-formulaic titles that represent niche content categories.
🚀 How to Run the App
Install dependencies:

Bash
pip install streamlit pandas scikit-learn nltk wordcloud
Launch the Streamlit app:

Bash
streamlit run app.py

How to use the live demo
- Open the live demo at: `https://netflixcapstone.streamlit.app/` or run locally with the command above.
- Paste a content description into the text area (examples below) and click **Predict & Recommend**.
- The app predicts a thematic cluster and shows 5 suggested titles from that cluster — each suggestion includes the poster image plus inline action links ("View on Netflix" / "IMDb") that open in a new tab.
- Use the **🔄 Suggest More** button to load additional unseen recommendations from the same cluster. When the cluster is exhausted the control becomes **🔁 Reset & Suggest Again**.

Quick input examples (paste into the description box)
- "Investigative true-crime documentary about organized crime and corruption"
- "Lighthearted family animation about friendship and adventurous animals"
- "Sci-fi action with space travel and time-manipulation themes"

Behaviour & matching
- Recommendations are filtered by semantic similarity to the description (TF‑IDF + cosine similarity). This prevents overly broad results — e.g. searching for a crime documentary will not return unrelated sports documentaries.
- If the model cannot find close matches inside the predicted cluster, the UI will show a warning and fall back to broader results.

Configuration & tuning
- Similarity threshold (controls result strictness): edit `SIM_THRESHOLD` inside `app.py` (default: `0.08`). Lower → broader results, Higher → stricter results.
- To show confidence scores or change the recommendation batch size, edit `pick_and_store()` in `app.py`.

Contact & support
- Create an issue on this repository for bugs or feature requests: https://github.com/Ansh1kaDixit/CapstoneNetflix/issues
- Author/GitHub: Ansh1kaDixit — https://github.com/Ansh1kaDixit
- For quick questions, open an issue and tag `@Ansh1kaDixit`.

Contributing
- Contributions are welcome — fork the repo, create a branch, add tests/docs, and open a pull request. Use Issues to discuss bigger changes first.

License
- This repository is provided for educational/demo purposes. Add a LICENSE file if you plan to publish or distribute the code.
📂 Repository Structure
netflix_ml.ipynb: Full Jupyter Notebook containing EDA, hypothesis testing, and model development.

app.py: Streamlit application script for real-time cluster prediction.

netflix_kmeans_model.pkl: Saved K-Means model for deployment.

netflix_tfidf_vectorizer.pkl: Saved TF-IDF vectorizer.

netflix_final_clustered_data.csv: The finalized dataset with cluster labels and integrated IMDb scores.

Anshika Dixit |