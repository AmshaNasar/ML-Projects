import pandas as pd

# Load ratings file (tab-separated)
ratings = pd.read_csv("u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])

print(ratings.head())

movies = pd.read_csv(
    "u.item",
    sep="|",
    names=["movieId", "title"],
    usecols=[0,1],
    encoding="latin-1"
)

# Merge ratings with movie titles
ratings = ratings.merge(movies, on="movieId")

print(ratings.head())

# Create user–item matrix
user_item_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")

print(user_item_matrix.head())

from sklearn.metrics.pairwise import cosine_similarity

# Fill NaN with 0, then transpose (so rows = movies)
item_matrix = user_item_matrix.fillna(0).T

# Cosine similarity between movies
similarity = cosine_similarity(item_matrix)

# Store in dataframe for easy lookup
item_similarity_df = pd.DataFrame(similarity, 
                                  index=item_matrix.index, 
                                  columns=item_matrix.index)

def recommend_movies(user_id, num_recommendations=5):
    # Movies the user has rated
    user_ratings = user_item_matrix.loc[user_id].dropna()
    scores = {}

    for movie_id, rating in user_ratings.items():
        # Similar movies
        similar_movies = item_similarity_df[movie_id].drop(movie_id)
        
        for sim_movie, sim_score in similar_movies.items():
            if sim_movie not in user_ratings.index:  # skip already watched
                scores[sim_movie] = scores.get(sim_movie, 0) + sim_score * rating

    # Rank by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Convert IDs to titles
    top_movies = [movies[movies.movieId == mid].title.values[0] for mid, _ in ranked[:num_recommendations]]
    
    return top_movies

# pick 10 users (first 10 or random 10)
sample_users = range(1, 11)   # users 1–10
# OR random choice:
# import random
# sample_users = random.sample(df['user_id'].unique().tolist(), 10)

# loop through them and print recommendations
for user_id in sample_users:
    recs = recommend_movies(user_id, num_recommendations=5)
    print(f"\nTop 5 Recommendations for User {user_id}:")
    for rank, movie in enumerate(recs, start=1):
        print(f"  {rank}. {movie}")



matrix_filled = user_item_matrix.fillna(0)

from sklearn.decomposition import TruncatedSVD
import numpy as np

svd = TruncatedSVD(n_components=20, random_state=42)  # 20 latent features
latent_matrix = svd.fit_transform(matrix_filled)

# Reconstruct approximation
approx_matrix = np.dot(latent_matrix, svd.components_)

# Convert back to dataframe (predicted ratings for all users & movies)
predicted_ratings = pd.DataFrame(
    approx_matrix, 
    index=matrix_filled.index, 
    columns=matrix_filled.columns
)

def recommend_movies_svd(user_id, num_recommendations=5):
    user_predictions = predicted_ratings.loc[user_id]
    already_rated = user_item_matrix.loc[user_id].dropna().index
    
    # Remove movies user already rated
    recommendations = user_predictions.drop(already_rated)
    
    # Top N
    top_movies = recommendations.nlargest(num_recommendations).index
    return movies[movies.movieId.isin(top_movies)].title.values


# Example: first 10 users (or random 10 if you prefer)
sample_users = range(1, 11)
# import random
# sample_users = random.sample(user_item_matrix.index.tolist(), 10)

for user_id in sample_users:
    recs = recommend_movies_svd(user_id, num_recommendations=5)
    print(f"\nTop 5 SVD Recommendations for User {user_id}:")
    for rank, movie in enumerate(recs, start=1):
        print(f"  {rank}. {movie}")


from sklearn.model_selection import train_test_split

def precision_at_k(user_id, k=5):
    # Movies the user actually rated
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    # Split into train/test
    train, test = train_test_split(user_ratings, test_size=0.2, random_state=42)
    
    # Build train matrix
    train_matrix = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Get recommendations
    recs = recommend_movies(user_id, train_matrix, similarity_matrix, k=k)
    
    # Actual relevant movies = movies rated >= 4 in test
    relevant = test[test['rating'] >= 4]['movieId'].values
    
    # Precision = fraction of recommended movies that are relevant
    hits = [movie for movie in recs if movie in relevant]
    precision = len(hits) / k
    
    return precision


def precision_at_k(user_id, k=5):
    # Movies user actually rated ≥4 (liked)
    actual = set(ratings.query("userId == @user_id and rating >= 4")["movieId"])
    
    # Top-K predicted movies from SVD
    predicted = set(predicted_ratings.loc[user_id].nlargest(k).index)
    
    return len(actual & predicted) / k

print("Precision@5 for User 1:", precision_at_k(1, k=5))

