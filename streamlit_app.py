from content_based_recommender import ContentBasedRecommender
import pandas as pd
import streamlit as st
import joblib

# Load datasets
@st.cache_data
def load_data():
    users = pd.read_csv("users.csv")
    ratings = pd.read_csv("ratings.csv")
    recipes = pd.read_csv("recipes.csv")
    return users, ratings, recipes

users, ratings, recipes = load_data()

# Load or Initialize Content-Based Recommender
@st.cache_resource
def load_model():
    try:
        return joblib.load("content_based_recommender_model.pkl")
    except Exception:
        return ContentBasedRecommender(
            recipe_df=recipes, 
            interactions_train_indexed_df=ratings.set_index("user_id"), 
            user_df=users
        )

content_model = load_model()

# Streamlit app logic
st.title("Diet Recommendation System")
user_id = st.text_input("Enter User ID:")

if user_id:
    try:
        user_id = int(user_id)
        recommendations = content_model.recommend_items(user_id, topn=10)
        if not recommendations.empty:
            st.table(recommendations)
        else:
            st.warning("No recommendations available.")
    except ValueError:
        st.error("Invalid User ID. Please enter a numeric value.")
else:
    st.info("Enter a User ID to get recommendations.")
