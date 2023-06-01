import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the restaurant data into a pandas DataFrame
data = pd.read_csv('01_MMla_with_reviews.csv')

# Preprocessing
# Clean the data (e.g., remove irrelevant columns)
data = data[['id', 'name', 'review_count', 'categories', 'rating', 'price', 'city', 'latitude', 'longitude']]

# Drop rows with missing values
data.dropna(inplace=True)

# Feature representation
# Convert restaurant names to feature vectors using TF-IDF vectorization
vectorizer = TfidfVectorizer()
name_features = vectorizer.fit_transform(data['name'])

# Calculate similarity
# Calculate cosine similarity between restaurant names
similarity_matrix = cosine_similarity(name_features, name_features)

# Define the Streamlit app
def main():
    st.title('Restaurant Recommender')
    
    # Dropdown menu for restaurant names
    restaurant_names = data['name'].tolist()
    selected_name = st.selectbox('Select a restaurant name:', restaurant_names)
    
    num_recommendations = st.slider('Number of recommendations:', 1, 10, 5)
    
    # Get recommendations based on selected name
    recommendations = get_recommendations(selected_name, num_recommendations)
    
    # Display the recommendations
    st.subheader('Recommended Restaurants:')
    st.table(recommendations[['name', 'categories', 'rating', 'price', 'city']])
    
# Function to get restaurant recommendations
def get_recommendations(restaurant_name, num_recommendations):
    # Get the index of the restaurant
    index = data[data['name'] == restaurant_name].index[0]
    
    # Get the similarity scores for the given restaurant
    similarity_scores = similarity_matrix[index]
    
    # Sort the restaurants based on similarity scores
    similar_restaurants_indices = similarity_scores.argsort()[::-1][1:]  # Exclude the current restaurant
    
    # Get the top similar restaurants
    top_similar_restaurants = data.iloc[similar_restaurants_indices][:num_recommendations]
    
    return top_similar_restaurants

# Run the Streamlit app
if __name__ == '__main__':
    main()
