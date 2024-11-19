import pandas as pd


class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, interactions_full_df, recipe_df):
        # Compute the most popular items based on ratings > 3
        interactions_filtered_df = interactions_full_df.loc[interactions_full_df['rating'] > 3]
        self.popularity_df = (
            interactions_filtered_df
            .groupby('recipe_id')['rating']
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        self.recipe_df = recipe_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, calorie_limit, items_to_ignore=[], topn=10):
     """Recommend items based on popularity and calorie limit."""
     # Filter items by calorie limit
     recommendations_df = self.popularity_df[
         ~self.popularity_df['recipe_id'].isin(items_to_ignore)
     ].sort_values('rating', ascending=False)
    
     recommendations_df = recommendations_df.merge(
         self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id'
     )
    
    # Filter by calorie limit
     recommendations_df = recommendations_df[recommendations_df['calories'] <= calorie_limit]

    # Group and prioritize by diet labels
     balanced_df = recommendations_df[recommendations_df['diet_labels'].str.contains('balanced', na=False)].head(2)
     highprotein_df = recommendations_df[recommendations_df['diet_labels'].str.contains('highprotein', na=False)].head(2)
     highfiber_df = recommendations_df[recommendations_df['diet_labels'].str.contains('highfiber', na=False)].head(2)
     lowcarb_df = recommendations_df[recommendations_df['diet_labels'].str.contains('lowcarb', na=False)].head(2)
     lowfat_df = recommendations_df[recommendations_df['diet_labels'].str.contains('lowfat', na=False)].head(2)
     lowsodium_df = recommendations_df[recommendations_df['diet_labels'].str.contains('lowsodium', na=False)].head(2)
    
     # Combine results and return
     combined_frame = [balanced_df, highprotein_df, highfiber_df, lowcarb_df, lowfat_df, lowsodium_df]
     recommendations_df = pd.concat(combined_frame).drop_duplicates().head(topn)
     return recommendations_df[['recipe_id', 'recipe_name', 'calories', 'diet_labels']]

