import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Create pivot table
final_dataset = ratings.pivot(index="movieId", columns="userId", values="rating")
final_dataset.fillna(0, inplace=True)

# Filter movies and users based on interaction thresholds
no_user_voted = ratings.groupby("movieId")['rating'].agg('count')
no_movies_voted = ratings.groupby("userId")['rating'].agg('count')

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Create sparse matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Search function
def search_movie(movie_name):
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, na=False)]
    if len(movie_list) > 0:
        return movie_list.iloc[0]['movieId']
    return None

# Recommendation function
def get_recommendation(movie_idx_in_final):
    distances, indices = knn.kneighbors(csr_data[movie_idx_in_final], n_neighbors=11)
    recommended_movies = []

    for i, idx in enumerate(indices.flatten()[1:]):  # Start from 1 to skip the input movie itself
        movie_id = final_dataset.iloc[idx]['movieId']
        movie = movies[movies['movieId'] == movie_id].iloc[0]
        recommended_movies.append({
            'Title': movie['title'],
            'Genre': movie['genres'],
            'MovieId': movie_id
        })

    return pd.DataFrame(recommended_movies)

# Streamlit Interface
st.set_page_config(page_title="Enhanced Movie Recommendation System", layout="wide")

# Title and Header with Description
st.markdown("<h1 style='text-align: center; color:#6A0DAD;'>🎬 Movie Recommendation System 🎬</h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; font-size: 18px;'>Welcome to the movie recommendation system! Enter a movie name or click a button to discover your next favorite film!</p>
""", unsafe_allow_html=True)

# Layout for titles in the same horizontal line
title_col1, title_col2 = st.columns([2, 3])  # Adjust widths of columns for title alignment

# "Search and Browse Movies" title in the first column
with title_col1:
    st.markdown("<h3 style='color: #2e8b57;'>🔍 Search and Browse Movies</h3>", unsafe_allow_html=True)

# "Quick Browse by Category" title in the second column
with title_col2:
    st.markdown("<h3 style='color: #2e8b57;'>🎬 Quick Browse by Category</h3>", unsafe_allow_html=True)

# Use Streamlit columns for alignment
search_col, gap_col, button_col = st.columns([2, 0.1, 3])  # Adjust width proportions for spacing

# Search bar in the first column
with search_col:
    movie_input = st.text_input("Enter a Movie Name", "")
    if st.button("Get Recommendations", key="recommend_button"):
        if movie_input:
            movie_idx = search_movie(movie_input)
            if movie_idx is not None:
                movie_idx_in_final = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
                recommendations = get_recommendation(movie_idx_in_final)
                st.write("### 🎥 Recommended Movies")
                st.dataframe(recommendations.style.set_table_styles([{
                    'selector': 'thead th',
                    'props': [('background-color', '#ff6347'), ('color', 'white'), ('font-weight', 'bold')]
                }]))  # Table styling
            else:
                st.write("Movie not found.")
        else:
            st.write("Please enter a movie name to proceed.")

# Buttons in the third column
with button_col:
    # Define popular movie categories for buttons
    popular_categories = ["Action", "Comedy", "Drama", "Horror", "Romance"]

    # Create a horizontal row of buttons
    col_widths = [1 for _ in popular_categories]  # Equal spacing for buttons
    category_columns = st.columns(col_widths)
    selected_category = None

    for i, category in enumerate(popular_categories):
        if category_columns[i].button(category, key=f"{category}_button"):
            selected_category = category
            break

    # Show movies for the selected category
    if selected_category:
        st.write(f"### 📽️ Movies in {selected_category} Category")
        category_filtered_movies = movies[movies['genres'].str.contains(selected_category, case=False, na=False)]

        if not category_filtered_movies.empty:
            st.dataframe(category_filtered_movies[['title', 'genres']].style.set_table_styles([{
                'selector': 'thead th',
                'props': [('background-color', '#ff6347'), ('color', 'white'), ('font-weight', 'bold')]
            }]))
        else:
            st.write(f"No movies found in the {selected_category} category.")
