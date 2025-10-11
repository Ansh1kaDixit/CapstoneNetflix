🎬 Netflix Unsupervised Machine Learning & EDA Project

An exploratory and unsupervised machine learning analysis of Netflix’s 2019 catalog — combining data wrangling, visualization, and clustering to uncover content patterns and business insights.

📊 Overview

This project explores the catalog of Netflix titles as of 2019 (sourced from Flixable
), focusing on:

The evolution of Movies vs TV Shows on the platform

Regional content distribution and dominant genres

Unsupervised content clustering based on metadata and text features

The analysis integrates exploratory data analysis (EDA) with unsupervised learning (K-Means & Hierarchical clustering) to uncover natural groupings in Netflix’s library and derive actionable insights for content strategy and recommendation systems.

🧠 Objectives

Perform data cleaning and feature engineering on Netflix metadata

Conduct EDA to understand trends by type, year, and country

Use TF-IDF on text fields (description, listed_in, cast) for semantic features

Apply dimensionality reduction (Truncated SVD) for efficient clustering

Run K-Means and Hierarchical Clustering to group similar content

Interpret clusters and translate findings into business insights

📂 Dataset Description

The dataset contains metadata for Netflix titles available as of 2019.

Column Name	Description
show_id	Unique identifier for each title
type	Movie or TV Show
title	Title of the content
director	Director name(s)
cast	Cast members
country	Country of production
date_added	Date the title was added to Netflix
release_year	Original release year
rating	Content maturity rating (e.g., PG-13, TV-MA)
duration	Runtime or number of seasons
listed_in	Genre categories
description	Short summary of the title
🧩 Technologies & Libraries Used

Python 3.x

Pandas – Data manipulation and preprocessing

NumPy – Numerical operations

Matplotlib / Seaborn – Visualizations

Scikit-learn – TF-IDF, SVD, Clustering (KMeans, Hierarchical)

NLTK – Text preprocessing

🚀 Project Workflow

Data Wrangling

Handled missing values and normalized categorical fields

Derived new features like content_age, year_added_numeric, and type_encoded

Exploratory Data Analysis (EDA)

Visualized content growth over years

Compared movie vs TV show trends

Analyzed ratings and top genres per region

Feature Engineering

Created text_blob combining multiple textual columns

Generated TF-IDF embeddings for semantic content similarity

Applied TruncatedSVD for dimensionality reduction

Unsupervised Learning

K-Means Clustering (optimal k via Elbow + Silhouette)

Hierarchical Clustering for interpretability check

Cluster interpretation through top terms and summary statistics

Insights

Netflix increasingly focuses on TV shows post-2015

Strong regional genre clusters (Indian dramas, US comedies, etc.)

Identified aging content clusters suitable for renewal or retirement decisions

💡 Key Findings

The number of movies on Netflix has declined since 2010, while TV shows nearly tripled.

The US, India, and UK dominate the catalog, with differing genre strengths.

Clusters reveal clear thematic patterns:

Cluster 0: Contemporary Netflix Originals & Dramas

Cluster 1: International & Family Shows

Cluster 2: Classic Hollywood Movies

🧭 Business Applications

Content Acquisition: Identify underrepresented genres or countries.

Personalization: Use cluster membership as recommendation features.

Catalog Management: Curate or retire underperforming clusters.

Regional Strategy: Align local catalog composition with cluster insights.

🎥 Presentation Resources

📓 Notebook: netfilxml_final.ipynb

🎤 Teleprompter Script: teleprompter_Anshika_Dixit.md
 — Ready for a 15–20 minute video explanation

🔮 Future Work

Integrate IMDB or Rotten Tomatoes ratings for quality validation

Add user engagement metrics (views, retention) for semi-supervised fine-tuning

Experiment with advanced embeddings (BERT or Sentence Transformers)

Automate cluster labeling using keyword summarization
