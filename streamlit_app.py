import pandas as pd
import streamlit as st
import joblib
import random
from content_based_recommender import ContentBasedRecommender
from popularity_recommender import PopularityRecommender
from cf_recommender import CFRecommender

# Load datasets
@st.cache_data
def load_data():
    users = pd.read_csv("users.csv")
    ratings = pd.read_csv("ratings.csv")
    recipes = pd.read_csv("recipes.csv")
    return users, ratings, recipes

users, ratings, recipes = load_data()

# Load Recommender Models
@st.cache_resource
def load_content_model():
    try:
        return joblib.load("content_based_recommender_model.pkl")
    except Exception:
        return ContentBasedRecommender(
            recipe_df=recipes, 
            interactions_train_indexed_df=ratings.set_index("user_id"), 
            user_df=users
        )

@st.cache_resource
def load_popularity_model():
    return PopularityRecommender(ratings, recipes)

# @st.cache_resource
# def load_collab_model():
#     return CFRecommender(
#         recipe_df=recipes, 
#         interactions_train_indexed_df=ratings.set_index("user_id"), 
#         user_df=users
#     )

# collab_model = load_collab_model()
content_model = load_content_model()
popularity_model = load_popularity_model()

# Function to calculate BMR
def calculate_bmr(weight, height, age, gender):
    if gender.lower() == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    elif gender.lower() == 'female':
        return 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Invalid gender. Please enter 'male' or 'female'.")

# Random Health Tips
health_tips = [
    "Drink at least 8 glasses of water daily to stay hydrated.",
    "Include leafy greens in your diet for essential vitamins and minerals.",
    "Get 7-8 hours of quality sleep to help your body recover and function well.",
    "Practice mindful eating—avoid distractions during meals.",
    "Exercise regularly, even a brisk walk counts!",
    "Cut down on sugary drinks—opt for water or herbal tea instead.",
    "Snack on nuts and seeds instead of processed junk food.",
    "Don’t skip breakfast; make it healthy and nutritious!"
]

# Multiplication factors for activity levels
activity_factors = {
    'sedentary': 1.2,
    'light active': 1.375,
    'moderately active': 1.55,
    'very active': 1.725,
    'extra active': 1.9
}

# Layout Enhancements
st.image("logo.png", use_container_width=True)  # Add your logo
st.title("🥗 Health Food and Diet Recommendation System")

# Create layout for interactive form
with st.form(key="user_form"):
    st.header("🔍 Get Your Personalized Recommendations")
    user_id_input = st.text_input("Enter Your User ID:")
    new_user = st.checkbox("I am a new user", value=False)

    if new_user:
        weight = st.number_input("Enter your weight (kg):", min_value=30.0, step=0.1)
        height = st.number_input("Enter your height (cm):", min_value=100.0, step=0.1)
        age = st.number_input("Enter your age:", min_value=10, step=1)
        gender = st.selectbox("Select your gender:", ['Male', 'Female'])
        activity_level = st.selectbox(
            "Select your activity level:",
            ['Sedentary', 'Light Active', 'Moderately Active', 'Very Active', 'Extra Active']
        )
    submit_button = st.form_submit_button(label="💡 Get Recommendations")

if submit_button:
    try:
        # Ensure user_id is numeric
        if not new_user:
            user_id = int(user_id_input)
        else:
            user_id = users['user_id'].max() + 1  # Assign a new user ID
        
        if user_id in users['user_id'].values or new_user:
            st.success("🎉 Generating Your Recommendations...")
            
            if new_user:
                # Calculate calorie intake
                bmr = calculate_bmr(weight, height, age, gender)
                calorie_limit = bmr * activity_factors[activity_level.lower()]
                st.write(f"Your calculated calorie limit is **{calorie_limit:.2f} kcal/day**.")

                # Use popularity-based recommendations for new users
                recommendations = popularity_model.recommend_items(
                    calorie_limit=calorie_limit / 7, items_to_ignore=[], topn=10
                )

            else:
                # Use collaborative filtering for existing users
                recommendations = content_model.recommend_items(user_id=user_id, topn=10)

            # Display recommendations
            if not recommendations.empty:
                st.subheader("🍽 Recommended Recipes for You")
                st.table(recommendations)
            else:
                st.warning("No recommendations available at the moment.")


            # Save the new user details
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

            # Show a random health tip
            st.info(f"💡 **Health Tip:** {random.choice(health_tips)}")
        else:
            st.error("Invalid User ID or User doesn't exist.")
    except ValueError:
        st.error("Please enter a valid numeric User ID.")

# Footer
st.markdown("---")
st.markdown("💡 *Stay healthy, eat well, and live your best life!*")
