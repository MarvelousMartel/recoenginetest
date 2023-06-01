import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# Load the dataset
data = pd.read_csv('01_MMla_with_reviews.csv')

# Perform one-hot encoding for the 'price_bucket' column
encoder = OneHotEncoder(sparse=False)
price_bucket_encoded = encoder.fit_transform(data['price_bucket'].values.reshape(-1, 1))
encoded_columns = encoder.categories_[0]

# Create the feature matrix
features = data[['rating', 'review_count']]
feature_matrix = pd.DataFrame(features.values, columns=features.columns)
feature_matrix = pd.concat([feature_matrix, pd.DataFrame(price_bucket_encoded, columns=encoded_columns)], axis=1)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)

# Define the function to get restaurant recommendations
def get_recommendations(restaurant_name, num_recommendations):
    # Check if the restaurant name exists in the dataset
    if restaurant_name not in data['name'].values:
        return []
    
    # Get the index of the restaurant with the given name
    index = data[data['name'] == restaurant_name].index[0]
    
    # Get the similarity scores for the given restaurant
    similarity_scores = similarity_matrix[index]
    
    # Sort the restaurants based on similarity scores
    sorted_indices = similarity_scores.argsort()[::-1]
    
    # Exclude the input restaurant itself
    sorted_indices = sorted_indices[1:]
    
    # Get the top N recommendations
    top_indices = sorted_indices[:num_recommendations]
    recommendations = data.loc[top_indices, 'name']
    
    return recommendations.tolist()

# Streamlit app
def main():
    st.title("Restaurant Recommender Engine")
    
    # Input restaurant name
    restaurant_name = st.selectbox("Select a restaurant:", data['name'].tolist())
    num_recommendations = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)
    
    # Get recommendations
    recommendations = get_recommendations(restaurant_name, num_recommendations)
    
    # Display recommendations
    if recommendations:
        st.subheader("Recommended Restaurants:")
        for i, recommendation in enumerate(recommendations):
            st.write(f"{i+1}. {recommendation}")
    else:
        st.write("No recommendations available for the selected restaurant.")

if __name__ == '__main__':
    main()
