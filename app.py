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

# Convert feature_matrix to a NumPy array
feature_matrix = np.asarray(feature_matrix)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)

# Define the function to get restaurant recommendations
def get_recommendations(restaurant_name, num_recommendations):
    # Get the index of the restaurant with the given name
    index = data[data['name'] == restaurant_name].index[0]

    # Get the similarity scores for the given restaurant
    similarity_scores = similarity_matrix[index]

    # Sort the restaurants based on similarity scores
    sorted_indices = similarity_scores.argsort()[::-1]
    
    # Exclude the restaurant itself from recommendations
    sorted_indices = sorted_indices[1:]

    # Get the top N recommendations
    top_indices = sorted_indices[:num_recommendations]

    # Get the restaurant names of the top recommendations
    recommendations = data.iloc[top_indices]['name'].values

    return recommendations

# Streamlit app
def main():
    st.title("Restaurant Recommendation System")
    
    # Dropdown for restaurant name selection
    restaurant_name = st.selectbox("Select a restaurant:", data['name'].values)

    num_recommendations = 5
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(restaurant_name, num_recommendations)
        st.subheader("Top Recommendations:")
        for i, recommendation in enumerate(recommendations):
            st.write(f"{i+1}. {recommendation}")

if __name__ == "__main__":
    main()
