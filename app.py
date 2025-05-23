import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load and preprocess data
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    movies['overview'] = movies['overview'].fillna('')
    movies['genres_text'] = movies['genres'].apply(lambda x: " ".join([g['name'] for g in ast.literal_eval(x)]))
    movies['combined'] = movies['genres_text'] + " " + movies['overview']
    return movies

# Build recommendation model
@st.cache_resource
def build_model(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title'].str.lower())
    return cosine_sim, indices

# Recommendation function
def recommend(title, cosine_sim, indices, movies, n=5):
    title = title.lower()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# App layout
st.title("🎬 Movie Recommender")
st.write("Get movie recommendations based on your favorite!")

# Load and model
movies = load_data()
cosine_sim, indices = build_model(movies)

# Dropdown for movie selection
selected_movie = st.selectbox("Select a movie:", sorted(movies['title'].unique()))

# Show recommendations
if st.button("Recommend"):
    recommendations = recommend(selected_movie, cosine_sim, indices, movies)
    if recommendations:
        st.subheader("You might also like:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("No recommendations found. Try another title.")
