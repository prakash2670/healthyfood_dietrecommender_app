import pandas as pd
import streamlit as st
import joblib
from content_based_recommender import ContentBasedRecommender
from popularity_recommender import PopularityRecommender
from cf_recommender import CFRecommender
import matplotlib.pyplot as plt
import random

# Set Streamlit page config at the very top
st.set_page_config(page_title="Health Food and Diet Recommendation", page_icon="🍏", layout="wide")

# Load datasets
@st.cache_data
def load_data():
    users = pd.read_csv("users.csv")
    ratings = pd.read_csv("ratings.csv")
    recipes = pd.read_csv("recipes.csv")
    return users, ratings, recipes

users, ratings, recipes = load_data()

# Load or Initialize Models
@st.cache_resource
def load_content_model():
    try:
        return joblib.load("content_based_recommender_model.pkl")
    except Exception:
        return ContentBasedRecommender(
            recipe_df=recipes,
            interactions_train_indexed_df=ratings.set_index("user_id"),
            user_df=users,
        )

@st.cache_resource
def load_popularity_model():
    return PopularityRecommender(ratings, recipes)

@st.cache_resource
def load_collab_model():
    return CFRecommender(
        recipe_df=recipes,
        interactions_train_indexed_df=ratings.set_index("user_id"),
        user_df=users,
    )

content_model = load_content_model()
popularity_model = load_popularity_model()

# Function to calculate BMR
def calculate_bmr(weight, height, age, gender):
    if gender.lower() == "male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    elif gender.lower() == "female":
        return 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Invalid gender. Please enter 'male' or 'female'.")

# User Profile
def update_user_profile(user_id, weight, height, age, gender, activity_level):
    height_inch = round(height / 2.54, 2)
    weight_lb = round(weight * 2.205, 2)
    height_mtr = height / 100
    bmi = round(weight / (height_mtr**2), 2)
    bmr = calculate_bmr(weight, height, age, gender)
    activity_factor = activity_factors[activity_level.lower()]
    calories_per_day = round(bmr * activity_factor, 2)
    return {
        "user_id": user_id,
        "Gender": gender,
        "Height_inch": height_inch,
        "Weight_lb": weight_lb,
        "height_mtr": height_mtr,
        "weight_kgs": weight,
        "BMI": bmi,
        "age": age,
        "activity": activity_level,
        "BMR": bmr,
        "calories_per_day": calories_per_day,
    }

# Multiplication factors for activity levels
activity_factors = {
    "sedentary": 1.2,
    "light active": 1.375,
    "moderately active": 1.55,
    "very active": 1.725,
    "extra active": 1.9,
}

# Streamlit UI
st.title("🍏 Health Food and Diet Recommendation System")

# Display Logo
st.image("logo.png", width=200)  # Replace with your logo's file path if necessary

# Adding a description with emojis
st.markdown(
    """
    Welcome to the **Health Food and Diet Recommendation System**! \n 🍽️ 
    Get personalized recommendations based on your profile and activity level. 📊
    """
)

tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "🧑‍⚕️ User Profile", "📝 Diet Tips"])

# User Input
with tab1:
    st.subheader("Enter Your Details to Get Recommendations")
    user_id_input = st.text_input("🔢 Enter User ID:", key="user_id_input")

    if user_id_input:
        try:
            user_id = int(user_id_input)

            if user_id in users["user_id"].values:
                st.success("✅ User found! Showing personalized recommendations...")
                recommendations = content_model.recommend_items(user_id=user_id, topn=10)
            else:
                st.warning("⚠️ New User Detected! Let's create your profile... 💡")
                weight = st.number_input("⚖️ Enter your weight (kg):", min_value=30.0, step=0.1)
                height = st.number_input("📏 Enter your height (cm):", min_value=100.0, step=0.1)
                age = st.number_input("🎂 Enter your age:", min_value=10, step=1)
                gender = st.selectbox("🚻 Select your gender:", ["Male", "Female"])
                activity_level = st.selectbox(
                    "💪 Select your activity level:",
                    ["Sedentary", "Light Active", "Moderately Active", "Very Active", "Extra Active"],
                )

                # Update user profile
                new_user_data = update_user_profile(user_id, weight, height, age, gender, activity_level)
                users = pd.concat([users, pd.DataFrame([new_user_data])], ignore_index=True)
                users.to_csv("users.csv", index=False)

                st.success("✅ New user profile updated successfully! 🎉")
                recommendations = popularity_model.recommend_items(
                    calorie_limit=new_user_data["calories_per_day"] / 7, items_to_ignore=[], topn=10
                )

            if not recommendations.empty:
                st.table(recommendations)
            else:
                st.warning("⚠️ No recommendations available.")
        except ValueError:
            st.error("❌ Invalid User ID. Please enter a numeric value.")

# User Profile Tab
with tab2:
    if user_id_input:
        try:
            user_id = int(user_id_input)  # Convert input to integer
            user_details = users[users["user_id"] == user_id]  # Filter user details

            if not user_details.empty:
                st.write("### 👤 User Profile")
                st.table(user_details)

                # Calorie distribution chart
                calorie_distribution = {
                    "Balanced Diet 🍽️": 50,
                    "Low Protein 🍗": 25,
                    "Low Fat 🥑": 15,
                    "High Carb 🍞": 10,
                }

                fig, ax = plt.subplots()
                ax.pie(
                    calorie_distribution.values(),
                    labels=calorie_distribution.keys(),
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.axis("equal")
                st.write("### 🥗 Diet Type Calorie Distribution")
                st.pyplot(fig)
            else:
                st.warning(f"⚠️ User ID {user_id} not found in the database. Please enter a valid User ID.")
        except ValueError:
            st.error("❌ Invalid User ID. Please enter a numeric value.")

# Diet Tips Tab
with tab3:
    tips = [
        "💧 Stay hydrated throughout the day.",
        "🍗 Include a mix of protein, carbs, and fats in each meal.",
        "🥦 Avoid processed foods and focus on whole, fresh ingredients.",
        "🍽️ Plan your meals ahead to stay consistent.",
        "⏰ Don't skip meals to maintain energy levels.",
    ]
    st.write("### 🌱 Random Diet Tip")
    # Adding a clickable button for more tips
    if st.button("Get Tip"):
        st.write(f"💡 {random.choice(tips)}")
