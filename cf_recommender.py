from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd

class CFRecommender:
    MODEL_NAME = 'Efficient On-Demand Collaborative SVD'
    NUMBER_OF_FACTORS_MF = 100

    def __init__(self, recipe_df=None, interactions_train_indexed_df=None, user_df=None):
        # Reset index to avoid errors during pivot
        interactions_train_indexed_df = interactions_train_indexed_df.reset_index()
        
        # Create the sparse matrix directly instead of a full pivot matrix
        self.users_items_sparse_matrix = csr_matrix(
            (interactions_train_indexed_df['rating'], 
             (interactions_train_indexed_df['user_id'], interactions_train_indexed_df['recipe_id'])),
            shape=(len(user_df), len(recipe_df))
        )

        self.users_ids = user_df['user_id'].values
        self.recipe_ids = recipe_df['recipe_id'].values
        self.recipe_df = recipe_df
        self.user_df = user_df
        print("Initialized recommender. Ready to process on demand.")

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        if user_id not in self.users_ids:
            print(f"User ID {user_id} not found in dataset.")
            return None

        # Get the user's row index in the sparse matrix
        user_index = np.where(self.users_ids == user_id)[0][0]
        user_ratings_sparse = self.users_items_sparse_matrix[user_index]

        # Perform SVD on the specific user row (low memory usage)
        svd = TruncatedSVD(n_components=self.NUMBER_OF_FACTORS_MF, algorithm='randomized')
        user_factors = svd.fit_transform(user_ratings_sparse)  # User-specific latent factors
        sigma = svd.singular_values_
        item_factors = svd.components_  # Latent factors for items

        # Predict ratings for all items for this user
        user_predicted_ratings = np.dot(user_factors, np.dot(np.diag(sigma), item_factors))

        # Normalize predictions
        user_predicted_ratings = (user_predicted_ratings - user_predicted_ratings.min()) / (
            user_predicted_ratings.max() - user_predicted_ratings.min()
        )

        # Create a DataFrame for predictions
        predictions_df = pd.DataFrame({
            'recipe_id': self.recipe_ids,
            'recStrength': user_predicted_ratings.flatten()
        }).sort_values(by='recStrength', ascending=False)

        # Exclude already interacted items
        recommendations_df = predictions_df[~predictions_df['recipe_id'].isin(items_to_ignore)].head(topn)

        # Merge with recipe details
        recommendations_df = recommendations_df.merge(
            self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id'
        )[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'calories', 'diet_labels']]

        # Apply calorie filtering
        recommendations_df = self.get_recommendation_for_user_calorie_count(recommendations_df, user_id)

        return recommendations_df

    def get_recommendation_for_user_calorie_count(self, cal_rec_df, user_id):
        # Retrieve the user's daily calorie requirement
        user_calories_per_day = self.user_df.loc[self.user_df['user_id'] == user_id, 'calories_per_day'].values
        if len(user_calories_per_day) == 0:
            return cal_rec_df  # If no calorie data, return the full recommendation DataFrame

        # Use one-third of daily calorie intake as a threshold
        user_calories = user_calories_per_day[0] / 3
        return cal_rec_df[cal_rec_df['calories'] <= user_calories]
