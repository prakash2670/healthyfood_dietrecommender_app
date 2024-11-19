{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "interactions_full_df = pd.read_csv('/content/drive/MyDrive/DATA/ratings.csv')\n",
        "recipe_df = pd.read_csv('/content/drive/MyDrive/DATA/recipes.csv')"
      ],
      "metadata": {
        "id": "6w_gmy9oT0VZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "diQadQsqSM2L"
      },
      "outputs": [],
      "source": [
        "class PopularityRecommender:\n",
        "    MODEL_NAME = 'Popularity'\n",
        "\n",
        "    def __init__(self, interactions_full_df, recipe_df=None):\n",
        "        # Computes the most popular items by grouping recipe_ids with maximum number of ratings and not just maximum ratings\n",
        "        interactions_full_df = interactions_full_df.loc[interactions_full_df['rating'] > 3]\n",
        "        self.popularity_df = interactions_full_df.groupby('recipe_id')['rating'].sum().sort_values(ascending=False).reset_index()\n",
        "        self.recipe_df = recipe_df\n",
        "\n",
        "    def get_model_name(self):\n",
        "        return self.MODEL_NAME\n",
        "\n",
        "    def recommend_items(self, items_to_ignore=[], topn=10, pd=None, newuser_cal_count=1000):\n",
        "        # Recommend the more popular items that the user hasn't seen yet.\n",
        "        recommendations_df = self.popularity_df[~self.popularity_df['recipe_id'].isin(items_to_ignore)].sort_values('rating', ascending=False)\n",
        "        recommendations_df = recommendations_df.merge(self.recipe_df, how='left', left_on='recipe_id', right_on='recipe_id')\n",
        "\n",
        "        # Get only those recommendations which have cal score < newuser_cal_count\n",
        "        recommendations_df = recommendations_df.loc[recommendations_df['calories'] <= newuser_cal_count]\n",
        "\n",
        "        # Get 2 recommendations for each diet label\n",
        "        balanced_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('balanced')].head(2)\n",
        "        highprotein_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('highprotein')].head(2)\n",
        "        highfiber_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('highfiber')].head(2)\n",
        "        lowcarb_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('lowcarb')].head(2)\n",
        "        lowfat_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('lowfat')].head(2)\n",
        "        lowsodium_df = recommendations_df.loc[recommendations_df['diet_labels'].str.contains('lowsodium')].head(2)\n",
        "\n",
        "        # Empty old DataFrame and combine the filtered recommendations\n",
        "        combined_frame = [balanced_df, highprotein_df, highfiber_df, lowcarb_df, lowfat_df, lowsodium_df]\n",
        "        recommendations_df = pd.concat(combined_frame)\n",
        "        recommendations_df = recommendations_df.sort_values('rating', ascending=False).head(topn)[['recipe_id', 'recipe_name', 'calories', 'diet_labels']]\n",
        "\n",
        "        return recommendations_df"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Initialize the recommender system\n",
        "recommender = PopularityRecommender(interactions_full_df, recipe_df)"
      ],
      "metadata": {
        "id": "JZ3OH_5fUrdc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Generate recommendations\n",
        "items_to_ignore = []  # List of items to exclude (already interacted with)\n",
        "topn = 10                # Number of recommendations to return\n",
        "newuser_cal_count = 300  # Maximum calorie count for recommendations\n",
        "\n",
        "# Get recommendations\n",
        "recommendations = recommender.recommend_items(\n",
        "    items_to_ignore=items_to_ignore,\n",
        "    topn=topn,\n",
        "    pd=pd,\n",
        "    newuser_cal_count=newuser_cal_count\n",
        ")\n",
        "\n",
        "# Display the recommendations\n",
        "print(recommendations)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tC_44v1pVTZ9",
        "outputId": "7b64fb7c-e509-41c3-fea2-33e4ad1668de"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "    recipe_id                                 recipe_name   calories  \\\n",
            "0       17652                        Banana Crumb Muffins  263.25940   \n",
            "1       56927               Delicious Ham and Potato Soup  194.87740   \n",
            "6       50644                    Broiled Tilapia Parmesan  224.38160   \n",
            "13      25037  Best Big, Fat, Chewy Chocolate Chip Cookie  284.70520   \n",
            "14      10549                               Best Brownies  182.66820   \n",
            "22       9023                      Baked Teriyaki Chicken  271.86990   \n",
            "25      24264                              Sloppy Joes II  188.82200   \n",
            "30      14231                                   Guacamole  261.50300   \n",
            "36      10687                  Mrs. Sigg's Snickerdoodles   92.16035   \n",
            "50      49552                      Quinoa and Black Beans  153.33740   \n",
            "\n",
            "            diet_labels  \n",
            "0              balanced  \n",
            "1              balanced  \n",
            "6           highprotein  \n",
            "13            lowsodium  \n",
            "14            lowsodium  \n",
            "22          highprotein  \n",
            "25  highprotein lowcarb  \n",
            "30            highfiber  \n",
            "36    lowcarb lowsodium  \n",
            "50     highfiber lowfat  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "-qyhstDbVZnQ"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}