import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
data = pd.read_csv('01_MMla_with_reviews.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Feature representation
# Create a feature matrix using review count, price, latitude, and longitude
features = data[['review_count', 'price', 'latitude', 'longitude']]
features['price'] = features['price'].astype(str)  # Convert price to string
feature_matrix = features.values

# Calculate cosine similarity
similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)

# Define the function to get restaurant recommendations
def get_recommendations(restaurant_name, num_recommendations):
    # Get the index of the restaurant with the given name
    index = data[data['name'] == restaurant_name].index[0]
    
    # Get the similarity scores for the given restaurant
    similarity_scores = similarity_matrix[index]
    
    # Get the indices of the top recommendations
    top_indices = similarity_scores.argsort()[-num_recommendations - 1:-1][::-1]
    
    # Get the names of the top recommendations
    top_recommendations = data.loc[top_indices, 'name']
    
    return top_recommendations.tolist()

# Streamlit app
st.title("Restaurant Recommendation Engine")

# Dropdown for selecting a restaurant
restaurant_name = st.selectbox("Select a restaurant:", data['name'].tolist())

# Number of recommendations
num_recommendations = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

# Get recommendations
recommendations = get_recommendations(restaurant_name, num_recommendations)

# Display recommendations
st.subheader("Top Recommendations:")
for i, recommendation in enumerate(recommendations):
    st.write(f"{i+1}. {recommendation}")
