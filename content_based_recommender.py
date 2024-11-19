import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')


class ContentBasedRecommender:
    MODEL_NAME = 'ContentBased'
    CB_SCORE_RATING_FACTOR = 4.0

    def __init__(self, recipe_df, interactions_train_indexed_df, user_df):
        # Validate inputs
        if recipe_df is None or interactions_train_indexed_df is None or user_df is None:
            raise ValueError("recipe_df, interactions_train_indexed_df, and user_df cannot be None")

        # TF-IDF vectorizer initialization
        vectorizer = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3), min_df=0.01, max_df=0.8,
            stop_words=stopwords.words('english')
        )

        # Prepare recipe IDs
        self.recipe_ids = recipe_df['recipe_id'].tolist()

        # Build TF-IDF matrix
        self.tfidf_matrix = vectorizer.fit_transform(
            recipe_df['cook_method'] + " " + recipe_df['ingredients'] + " " + recipe_df['diet_labels']
        )
        self.tfidf_feature_names = vectorizer.get_feature_names_out()

        # Store necessary attributes
        self.recipe_df = recipe_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.user_df = user_df

        # Initialize a user profile cache
        self.user_profiles = {}

        print("\nTF-IDF matrix built with {} features.".format(len(self.tfidf_feature_names)))

    def get_model_name(self):
        return self.MODEL_NAME

    def get_item_profile(self, item_id):
        idx = self.recipe_ids.index(item_id)
        return self.tfidf_matrix[idx:idx + 1]

    def get_item_profiles(self, ids):
        item_profiles_list = [self.get_item_profile(x) for x in ids]
        return sparse.vstack(item_profiles_list)

    def build_user_profile(self, user_id):
        if user_id not in self.interactions_train_indexed_df.index:
            return None

        interactions_person_df = self.interactions_train_indexed_df.loc[user_id]
        user_items = (
            interactions_person_df['recipe_id'].tolist()
            if isinstance(interactions_person_df['recipe_id'], pd.Series)
            else [interactions_person_df['recipe_id']]
        )
        user_item_profiles = self.get_item_profiles(user_items)
        user_item_strengths = np.array(interactions_person_df['rating']).reshape(-1, 1)

        if np.sum(user_item_strengths) == 0:
            return None

        user_profile = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0)
        return normalize(np.asarray(user_profile).flatten().reshape(1, -1))

    def _get_similar_items_to_user_profile(self, user_id):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = self.build_user_profile(user_id)

        user_profile = self.user_profiles.get(user_id)
        if user_profile is None:
            return None

        cosine_similarities = linear_kernel(user_profile, self.tfidf_matrix)
        similar_indices = cosine_similarities.argsort().flatten()[::-1]

        return [(self.recipe_ids[i], cosine_similarities[0, i]) for i in similar_indices]

    def get_recommendation_for_user_calorie_count(self, recommendations_df, user_id):
        user_calories_per_day = self.user_df.loc[self.user_df['user_id'] == user_id, 'calories_per_day'].values
        if len(user_calories_per_day) == 0:
            return recommendations_df

        user_calories = user_calories_per_day[0] / 3
        return recommendations_df[recommendations_df['calories'] <= user_calories]

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        if similar_items is None:
            return pd.DataFrame()

        filtered_items = [item for item in similar_items if item[0] not in items_to_ignore]
        recommendations_df = pd.DataFrame(filtered_items, columns=['recipe_id', 'recStrength']).head(topn)

        recommendations_df = recommendations_df.merge(
            self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id'
        )[['recStrength', 'recipe_id', 'recipe_name', 'ingredients', 'calories', 'diet_labels']]

        return self.get_recommendation_for_user_calorie_count(recommendations_df, user_id)
