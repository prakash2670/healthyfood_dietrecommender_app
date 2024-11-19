import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load datasets
@st.cache
def load_data():
    users = pd.read_csv("users.csv")
    ratings = pd.read_csv("ratings.csv")
    recipes = pd.read_csv("receipe.csv")
    return users, ratings, recipes

users, ratings, recipes = load_data()

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    content_model = joblib.load("content_based_model.pkl")
    collab_model = joblib.load("collaborative_model.pkl")
    hybrid_model = joblib.load("hybrid_model.pkl")
    return content_model, collab_model, hybrid_model

content_model, collab_model, hybrid_model = load_models()

# Helper functions
def calculate_bmr(weight, height, age, gender):
    """Calculate BMR using Mifflin-St Jeor Equation."""
    if gender == "Male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def get_calorie_multiplier(activity_level):
    """Return the calorie multiplication factor based on activity level."""
    activity_factors = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Extra Active": 1.9
    }
    return activity_factors.get(activity_level, 1.2)

# Streamlit interface
st.title("Diet Recommendation System")
st.write("Personalized diet recommendations based on your profile and preferences.")

# Input user ID
user_id = st.text_input("Enter User ID (leave blank if new user):")

if user_id:
    if int(user_id) in users["user_id"].values:
        # Existing user flow
        st.write("Welcome back! Fetching recommendations...")
        # Collaborative recommendations
        collab_recommendations = collab_model.recommend(user_id, ratings, recipes)
        # Content-based recommendations
        content_recommendations = content_model.recommend(user_id, ratings, recipes)
        # Hybrid recommendations
        hybrid_recommendations = hybrid_model.recommend(
            user_id, collab_recommendations, content_recommendations, recipes
        )

        st.write("**Top Recommendations for You:**")
        st.table(hybrid_recommendations)

    else:
        st.error("User ID not found. Please register as a new user below.")
else:
    # New user registration flow
    st.write("Welcome, new user! Please provide your details.")
    weight = st.number_input("Weight (kg):", min_value=0.0, step=0.1)
    height = st.number_input("Height (cm):", min_value=0.0, step=0.1)
    age = st.number_input("Age:", min_value=0, step=1)
    gender = st.selectbox("Gender:", ["Male", "Female"])
    activity_level = st.selectbox(
        "Activity Level:",
        ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"]
    )

    if st.button("Submit"):
        # Calculate BMR and required calorie intake
        bmr = calculate_bmr(weight, height, age, gender)
        calorie_multiplier = get_calorie_multiplier(activity_level)
        required_calories = bmr * calorie_multiplier
        st.write(f"Your required calorie intake is approximately **{required_calories:.2f} kcal/day**.")

        # Save new user data
        new_user_id = users["user_id"].max() + 1
        new_user = pd.DataFrame([[new_user_id, weight, height, age, gender]], 
                                columns=["user_id", "weight", "height", "age", "gender"])
        users = pd.concat([users, new_user], ignore_index=True)
        users.to_csv("users.csv", index=False)
        st.success("Your profile has been created!")

        # Popularity-based recommendations
        st.write("**Recommended Foods Based on Popularity:**")
        popular_recipes = recipes[
            (recipes["calories"] <= required_calories) & 
            (recipes["ratings"] >= 3)
        ]
        st.table(popular_recipes[["recipe_id", "name", "calories", "label", "ratings"]])

