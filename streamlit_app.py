import pandas as pd
import streamlit as st
import joblib
from content_based_recommender import ContentBasedRecommender
from popularity_recommender import PopularityRecommender
from cf_recommender import CFRecommender  # Collaborative Filtering Recommender
from PIL import Image
import matplotlib.pyplot as plt
import random

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

# Load or Initialize Popularity-Based Recommender
@st.cache_resource
def load_popularity_model():
    return PopularityRecommender(ratings, recipes)

# # Load or Initialize Collaborative Filtering Model
# @st.cache_resource
# def load_collab_model():
#     return CFRecommender(
#         recipe_df=recipes, 
#         interactions_train_indexed_df=ratings.set_index("user_id"), 
#         user_df=users
#     )

content_model = load_content_model()
popularity_model = load_popularity_model()
# collab_model = load_collab_model()

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

# Health Tips
health_tips = [
    "🍎 An apple a day keeps the doctor away!",
    "💧 Drink at least 8 glasses of water daily.",
    "🏃‍♂️ 30 minutes of exercise can boost your mood and health.",
    "🥗 Include greens in every meal for better digestion.",
    "🛌 A good night's sleep is key to a healthy life.",
    "🍳 Cook meals at home to control ingredients and portions."
]

# UI Enhancements
logo = Image.open("logo.png")  # Replace with your logo file
st.image(logo, use_column_width=True)
st.title("🌟 Health Food and Diet Recommendation System 🌟")
st.markdown("Your personalized health companion for smarter food choices!")

# Sidebar for User Input
st.sidebar.header("User Input")
user_id_input = st.sidebar.text_input("Enter User ID:", placeholder="e.g., 12345")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🍲 Recommendations", "🧍 Profile", "💡 Tips"])

if user_id_input:
    try:
        # Ensure user_id is an integer
        user_id = int(user_id_input)

        # Check if the user_id exists in the dataset
        if user_id in users['user_id'].values:
            st.sidebar.success("Welcome back, User!")
            
            # Collaborative Filtering Recommendations
            with tab1:
                recommendations = content_model.recommend_items(user_id=user_id, topn=10)
                if not recommendations.empty:
                    st.success("Here are your personalized recommendations!")
                    st.table(recommendations)
                else:
                    st.warning("No recommendations available.")
            
        else:
            st.sidebar.warning("New User Detected! Fill in your details below.")
            
            # New User Input Form
            with st.sidebar:
                weight = st.number_input("Enter your weight (kg):", min_value=30.0, step=0.1)
                height = st.number_input("Enter your height (cm):", min_value=100.0, step=0.1)
                age = st.number_input("Enter your age:", min_value=10, step=1)
                gender = st.radio("Select your gender:", ['Male', 'Female'])
                activity_level = st.selectbox(
                    "Select your activity level:",
                    ['Sedentary', 'Light Active', 'Moderately Active', 'Very Active', 'Extra Active']
                )
            
            # Calculate BMR and Calorie Intake
            bmr = calculate_bmr(weight, height, age, gender)
            calorie_limit = bmr * activity_factors[activity_level.lower()]
            st.sidebar.write(f"Your calculated calorie limit is **{calorie_limit:.2f} kcal/day**.")

            # Popularity-Based Recommendations
            with tab1:
                recommendations = popularity_model.recommend_items(
                    calorie_limit=calorie_limit / 7, items_to_ignore=[], topn=10
                )
                if not recommendations.empty:
                    st.success("Here are some popular choices for you!")
                    st.table(recommendations)
                else:
                    st.warning("No recommendations available.")
            
            # Save new user details
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
            st.sidebar.success("New user details saved successfully!")

        # Show Random Tip
        with tab3:
            st.info(f"💡 **Health Tip of the Day:** {random.choice(health_tips)}")

        # User Profile
        with tab2:
            user_details = users[users['user_id'] == user_id]
            if not user_details.empty:
                st.write("### User Details")
                st.table(user_details)
            
            # Pie Chart for Calorie Allocation
            calorie_data = [calorie_limit / 3, calorie_limit / 3, calorie_limit / 3]
            labels = ['Breakfast', 'Lunch', 'Dinner']

            fig, ax = plt.subplots()
            ax.pie(calorie_data, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
            st.write("### Recommended Calorie Distribution")
            st.pyplot(fig)

    except ValueError:
        st.sidebar.error("Invalid User ID. Please enter a numeric value.")
else:
    st.info("Enter a User ID in the sidebar to get recommendations.")
