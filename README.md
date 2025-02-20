# 🎈 Health Recommender app ( Streamlit Deployed Version )
# HealthyFood Diet Recommender App

## Overview
The HealthyFood Diet Recommender App is a machine learning-based application designed to provide personalized diet recommendations based on user profiles and preferences. The app leverages public datasets containing user ratings and recipes to offer tailored diet plans. For new users, the app calculates BMI using height and weight inputs and recommends diets accordingly. For returning users, it utilizes collaborative filtering and content-based filtering to suggest diets based on their previous reviews and preferences.

## Features
- **BMI Calculation**: The app calculates the user's BMI using their height and weight.
- **Personalized Recommendations**: 
  - For new users, diet recommendations are based on BMI.
  - For returning users, recommendations are based on previous reviews and preferences.
- **Machine Learning Models**: Utilizes collaborative filtering and content-based filtering for accurate and personalized recommendations.
- **Public Datasets**: Trained on publicly available datasets containing user ratings and recipes.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/prakash2670/healthyfood_dietrecommender_app.git
   ```
2. Navigate
   ```bash
   cd healthyfood_dietrecommender_app
   ```
3. Install Requirements
   ```bash
   pip install -r requirements.txt
   ```
4.Run the app
   ```bash
   python app.py
   ```

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)


Usage
New Users: Enter your height and weight to calculate BMI and receive diet recommendations.

Returning Users: Log in to receive diet recommendations based on your previous reviews and preferences.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.


Acknowledgments
Public datasets used for training the models.

Contributors and maintainers of the libraries and frameworks used in this project.
