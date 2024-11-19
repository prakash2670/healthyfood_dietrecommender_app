import pandas as pd
import streamlit as st
import joblib
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

# Load or Initialize Content-Based Recommender
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

def load_collab_model():
    return CFRecommender(
        recipe_df=recipes, 
        interactions_train_indexed_df=ratings.set_index("user_id"), 
        user_df=users
    )

# Load or Initialize Popularity-Based Recommender
@st.cache_resource
def load_popularity_model():
    return PopularityRecommender(ratings, recipes)



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

# Streamlit UI
st.title("Health Food and Diet Recommendation System")
user_id = st.text_input("Enter User ID:")

if user_id:
    try:
        user_id = int(user_id)
        if user_id in users['user_id'].values:
            # If user exists, use content-based recommendations
            # content_model = load_content_model()
            content_model = load_collab_model()
            recommendations = content_model.recommend_items(user_id=user_id, topn=10)
        else:
            # If user doesn't exist, ask for input and calculate BMR, then use popularity-based recommendations
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

            
            popularity_model = load_popularity_model()
            # Use popularity-based recommendations
            recommendations = popularity_model.recommend_items(
                calorie_limit=calorie_limit / 7, items_to_ignore=[], topn=10
            )


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

        if not recommendations.empty:
            st.table(recommendations)
        else:
            st.warning("No recommendations available.")

    except ValueError:
        st.error("Invalid User ID. Please enter a numeric value.")
else:
    st.info("Enter a User ID to get recommendations.")
