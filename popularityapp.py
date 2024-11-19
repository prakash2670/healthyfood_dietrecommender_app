import pandas as pd
import numpy as np
import streamlit as st
from content_recommender import ContentBasedRecommender
from popularity_recommender import PopularityRecommender

# Load datasets
@st.cache_data
def load_data():
    users = pd.read_csv("users.csv")
    ratings = pd.read_csv("ratings.csv")
    recipes = pd.read_csv("recipes.csv")
    return users, ratings, recipes

users, ratings, recipes = load_data()

# Initialize Recommenders
content_recommender = ContentBasedRecommender(recipes, ratings, users)
popularity_recommender = PopularityRecommender(ratings, recipes)

# Function to calculate BMR
def calculate_bmr(weight, height, age, gender):
    if gender.lower() == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    elif gender.lower() == 'female':
        return 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Invalid gender. Please enter 'male' or 'female'.")

# Multiplication factors for activity levels
activity_factors = {
    'sedentary': 1.2,
    'light active': 1.375,
    'moderately active': 1.55,
    'very active': 1.725,
    'extra active': 1.9
}

# Hybrid Recommendation System
def hybrid_recommend(user_id, calorie_limit, topn=10):
    """Combines recommendations from content and popularity recommenders."""
    # Content-based recommendations
    content_recs = content_recommender.recommend_items(user_id=user_id, topn=topn)
    
    # Popularity-based recommendations
    popularity_recs = popularity_recommender.recommend_items(
        calorie_limit=calorie_limit, items_to_ignore=[], topn=topn
    )
    
    # Merge recommendations by matching recipe_id and prioritize high ratings
    hybrid_recs = content_recs.merge(
        popularity_recs, on='recipe_id', suffixes=('_content', '_popularity')
    )
    hybrid_recs = hybrid_recs.sort_values(by=['recStrength_content', 'recStrength_popularity'], ascending=False)
    return hybrid_recs.head(topn)[['recipe_id', 'recipe_name', 'calories', 'diet_labels']]

# Streamlit UI
st.title("Hybrid Diet Recommendation System")
user_id = st.text_input("Enter User ID:")

if user_id:
    try:
        user_id = int(user_id)
        if user_id in users['user_id'].values:
            st.write(f"Welcome back, User {user_id}!")
            calorie_limit = users.loc[users['user_id'] == user_id, 'calories_per_day'].values[0]
            
            # Generate Hybrid Recommendations
            st.subheader("Hybrid Recommendations")
            recommendations = hybrid_recommend(user_id=user_id, calorie_limit=calorie_limit / 3, topn=10)
            if not recommendations.empty:
                st.table(recommendations)
            else:
                st.warning("No recommendations available for your preferences.")
        else:
            st.warning("User ID not found. Please provide additional details.")
            
            # Input new user details
            weight = st.number_input("Enter your weight (kg):", min_value=30.0, step=0.1)
            height = st.number_input("Enter your height (cm):", min_value=100.0, step=0.1)
            age = st.number_input("Enter your age:", min_value=10, step=1)
            gender = st.selectbox("Select your gender:", ['Male', 'Female'])
            activity_level = st.selectbox(
                "Select your activity level:",
                ['Sedentary', 'Light Active', 'Moderately Active', 'Very Active', 'Extra Active']
            )
            
            # Calculate BMR and calorie intake
            bmr = calculate_bmr(weight, height, age, gender)
            calorie_limit = bmr * activity_factors[activity_level.lower()]
            st.write(f"Your calculated calorie limit is {calorie_limit:.2f} kcal/day.")
            
            

            # Generate Popularity-Based Recommendations
            st.subheader("Popularity-Based Recommendations")
            recommendations = popularity_recommender.recommend_items(
                calorie_limit=calorie_limit / 7, items_to_ignore=[], topn=10
            )
            if not recommendations.empty:
                st.table(recommendations)
            else:
                st.warning("No recommendations available for your calorie limit.")


            # Save new user details to users.csv
            new_user_data = {
                'user_id': user_id,
                'weight': weight,
                'height': height,
                'age': age,
                'gender': gender,
                'calories_per_day': calorie_limit
            }
            users = pd.concat([users, pd.DataFrame([new_user_data])], ignore_index=True)
            users.to_csv("users.csv", index=False)
            st.success("New user details saved successfully!")

    except ValueError:
        st.error("Invalid User ID. Please enter a numeric value.")
else:
    st.info("Enter a User ID to get recommendations.")
