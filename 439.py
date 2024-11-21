import polars as pl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy
from scipy.sparse.linalg import svds
import math
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model


# Load data with Polars
def load_data(file_path):
    return pl.read_csv(file_path)

# Preprocess data: Train-test split
def preprocess_data(ratings, test_size=0.2):
    unique_users = ratings['userId'].unique()
    train_rows = []
    test_rows = []
    
    for user in unique_users:
        user_ratings = ratings.filter(pl.col('userId') == user)
        train_count = int((1 - test_size) * user_ratings.height)
        train_rows.append(user_ratings[:train_count])
        test_rows.append(user_ratings[train_count:])
    
    train_data = pl.concat(train_rows)
    test_data = pl.concat(test_rows)
    return train_data, test_data

# Create a user-item matrix using Polars
def create_user_item_matrix(data, movies, tags):
    data = data.to_dicts()
    users, items, ratings = zip(*[(d['userId'], d['movieId'], d['rating']) for d in data])
    user_item_matrix = {}
    
    # Extract genres and tags information
    movies_dict = movies.to_dict(as_series=False)
    tags_dict = tags.to_dict(as_series=False)

    movie_genres = {movies_dict['movieId'][i]: movies_dict['genres'][i].split('|') for i in range(len(movies_dict['movieId']))}
    user_tags = {}
    
    for i in range(len(tags_dict['userId'])):
        user_id = tags_dict['userId'][i]
        movie_id = tags_dict['movieId'][i]
        tag = tags_dict['tag'][i]
        if user_id not in user_tags:
            user_tags[user_id] = {}
        if movie_id not in user_tags[user_id]:
            user_tags[user_id][movie_id] = []
        user_tags[user_id][movie_id].append(tag)
    
    # Create user-item ratings
    for user, item, rating in zip(users, items, ratings):
        if user not in user_item_matrix:
            user_item_matrix[user] = {}
        user_item_matrix[user][item] = rating
    
    unique_users = sorted(set(users))
    unique_items = sorted(set(items))
    
    # Genre Encoding using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movie_genres.values())
    genre_labels = list(movie_genres.keys())
    genre_dict = {genre_labels[i]: genre_matrix[i] for i in range(len(genre_labels))}
    
    # Create a 2D matrix with rows as users and columns as items
    matrix = np.zeros((len(unique_users), len(unique_items)))
    for user_idx, user in enumerate(unique_users):
        for item_idx, item in enumerate(unique_items):
            rating = user_item_matrix.get(user, {}).get(item, 0)
            genre_vector = genre_dict.get(item, np.zeros(genre_matrix.shape[1]))
            tag_score = len(user_tags.get(user, {}).get(item, []))
            # Combine rating, genre preference, and tag score
            combined_score = rating + 0.1 * np.sum(genre_vector) + 0.05 * tag_score
            matrix[user_idx, item_idx] = combined_score
            
    return matrix, unique_users, unique_items

# Matrix Factorization (SVD)
def svd_recommendation(user_item_matrix, num_factors=50):
    U, sigma, Vt = svds(user_item_matrix, k=num_factors)
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    return predicted_ratings

# User-based Collaborative Filtering
def user_based_cf(train_data, unique_users, unique_items):
    # Create user-item matrix
    user_item_matrix = np.zeros((len(unique_users), len(unique_items)))

    # Fill the matrix with ratings
    for row in train_data.iter_rows(named=True):
        user_idx = unique_users.index(row['userId'])
        item_idx = unique_items.index(row['movieId'])
        user_item_matrix[user_idx, item_idx] = row['rating']

    # Compute user-user similarity matrix using cosine similarity
    user_similarity = np.corrcoef(user_item_matrix)

    # Predict ratings
    predicted_ratings = np.zeros_like(user_item_matrix)
    for user_idx in range(len(unique_users)):
        # For each user, predict the rating for each item
        for item_idx in range(len(unique_items)):
            if user_item_matrix[user_idx, item_idx] == 0:  # Predict only for unrated items
                # Weighted average of ratings from similar users
                numerator = np.dot(
                    user_similarity[user_idx],
                    user_item_matrix[:, item_idx]
                )
                denominator = np.sum(np.abs(user_similarity[user_idx]))
                predicted_ratings[user_idx, item_idx] = numerator / denominator if denominator != 0 else 0

    return predicted_ratings

# Generate recommendations for each user
def generate_recommendations(predicted_ratings, unique_users, unique_items, train_data, top_n=10):
    train_items = train_data.to_dict(as_series=False)
    recommendations = {}
    
    for user_id in unique_users:
        user_idx = unique_users.index(user_id)
        user_predictions = predicted_ratings[user_idx]
        rated_items = set(train_items['movieId'][i] for i in range(len(train_items['userId'])) if train_items['userId'][i] == user_id)
        
        unrated_items = [i for i in range(len(unique_items)) if unique_items[i] not in rated_items]
        top_items = sorted(unrated_items, key=lambda x: -user_predictions[x])[:top_n]
        recommended_item_ids = [unique_items[i] for i in top_items]
        recommendations[user_id] = recommended_item_ids

    return recommendations

# Evaluate recommendations with Precision, Recall, F-Measure, and NDCG
def evaluate_recommendations(recommendations, test_data, top_n=10):
    precision_list, recall_list, ndcg_list = [], [], []

    test_items = test_data.to_dict(as_series=False)
    for user_id, recommended_items in recommendations.items():
        relevant_items = set(
            item for item, user in zip(test_items['movieId'], test_items['userId']) if user == user_id
        )
        hits = len(set(recommended_items) & relevant_items)
        precision = hits / top_n if top_n > 0 else 0
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)

        # NDCG Calculation
        dcg = sum(
            1 / math.log2(idx + 2) for idx, item in enumerate(recommended_items) if item in relevant_items
        )
        idcg = sum(1 / math.log2(idx + 2) for idx in range(min(len(relevant_items), top_n)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)
    f_measure = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )
    return avg_precision, avg_recall, f_measure, avg_ndcg

# Evaluation metrics: MAE and RMSE
def evaluate(predictions, actual, mask):
    predictions = predictions[mask]
    actual = actual[mask]
    mae = mean_absolute_error(actual, predictions)
    rmse = math.sqrt(mean_squared_error(actual, predictions))
    return mae, rmse


###OUR APPROACH
# Proprietary Algorithm: Feature-Rich Neural Matrix Factorization (FR-NMF)
# Proprietary Algorithm: Feature-Rich Neural Matrix Factorization (FR-NMF)
def feature_rich_nmf(train_data, movies, tags, unique_users, unique_items, num_factors=50, epochs=10, batch_size=64):
    # Encoding user IDs and item IDs
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    train_data_dict = train_data.to_dict(as_series=False)
    user_ids = user_encoder.fit_transform(train_data_dict['userId'])
    item_ids = item_encoder.fit_transform(train_data_dict['movieId'])
    ratings = np.array(train_data_dict['rating'])

    num_users = len(unique_users)
    num_items = len(unique_items)
    
    # Extract genres and create a mapping of movieId to index in the genre_encoded array
    movie_genres = {row['movieId']: row['genres'].split('|') for row in movies.to_dicts()}
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(movie_genres.values())
    genre_labels = list(movie_genres.keys())
    genre_mapping = {genre_labels[i]: i for i in range(len(genre_labels))}

    genre_input_dim = genre_encoded.shape[1]

    # Model Definition
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    genre_input = Input(shape=(genre_input_dim,))

    user_embedding = Embedding(num_users, num_factors)(user_input)
    item_embedding = Embedding(num_items, num_factors)(item_input)

    user_vector = Flatten()(user_embedding)
    item_vector = Flatten()(item_embedding)

    # Concatenate user, item, and genre vectors
    concatenated = Concatenate()([user_vector, item_vector, genre_input])
    dense_1 = Dense(128, activation='relu')(concatenated)
    dense_2 = Dense(64, activation='relu')(dense_1)
    dense_3 = Dense(32, activation='relu')(dense_2)
    output = Dense(1)(dense_3)

    model = Model(inputs=[user_input, item_input, genre_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Preparing input data
    genre_features = []
    for movie_id in train_data_dict['movieId']:
        if movie_id in genre_mapping:
            genre_features.append(genre_encoded[genre_mapping[movie_id]])
        else:
            genre_features.append(np.zeros(genre_input_dim))  # If no genre info is available, use zeros

    genre_features = np.array(genre_features)

    # Training the model
    model.fit([user_ids, item_ids, genre_features], ratings, epochs=epochs, batch_size=batch_size, verbose=1)

    # Predicting the full user-item rating matrix
    all_user_ids = np.arange(num_users)
    all_item_ids = np.arange(num_items)

    # Predict ratings for all user-item pairs
    user_grid, item_grid = np.meshgrid(all_user_ids, all_item_ids)
    user_grid_flat = user_grid.flatten()
    item_grid_flat = item_grid.flatten()
    genre_grid_flat = np.array([genre_encoded[genre_mapping.get(unique_items[i], 0)] for i in item_grid_flat])

    predicted_ratings = model.predict([user_grid_flat, item_grid_flat, genre_grid_flat])
    predicted_ratings_matrix = predicted_ratings.reshape((num_users, num_items))

    return predicted_ratings_matrix


# Main function
def main():
    # Load data
    ratings = load_data('ratings.csv')
    movies = load_data('movies.csv')
    tags = load_data('tags.csv')

    # Preprocess data
    print("Preprocessing data...")
    train_data, test_data = preprocess_data(ratings)

    # Create user-item matrix
    print("Creating user-item matrix...")
    user_item_matrix, users, items = create_user_item_matrix(train_data, movies, tags)

    # SVD Recommendation
    print("Performing SVD...")
    predicted_ratings_svd = svd_recommendation(user_item_matrix)

    # Evaluate SVD predictions
    print("Evaluating SVD predictions...")
    mask = user_item_matrix > 0
    mae_svd, rmse_svd = evaluate(predicted_ratings_svd, user_item_matrix, mask)
    print(f"SVD Evaluation - MAE: {mae_svd:.4f}, RMSE: {rmse_svd:.4f}")

    # Generate recommendations using SVD
    print("Generating recommendations using SVD...")
    recommendations_svd = generate_recommendations(predicted_ratings_svd, users, items, train_data)

    # Evaluate recommendations using SVD
    print("Evaluating recommendations using SVD...")
    precision_svd, recall_svd, f_measure_svd, ndcg_svd = evaluate_recommendations(recommendations_svd, test_data)
    print(
        f"SVD Recommendation Evaluation - Precision: {precision_svd:.4f}, Recall: {recall_svd:.4f}, "
        f"F-Measure: {f_measure_svd:.4f}, NDCG: {ndcg_svd:.4f}"
    )

    # User-Based Collaborative Filtering - Secondary Check
    print("__________________________________________")
    print("USER-BASED CF: SECONDARY CHECK")
    
    print("Performing User-Based Collaborative Filtering...")
    predicted_ratings_cf = user_based_cf(train_data, users, items)

    # Evaluate CF predictions
    print("Evaluating User-Based CF predictions...")
    mae_cf, rmse_cf = evaluate(predicted_ratings_cf, user_item_matrix, mask)
    print(f"User-Based CF Evaluation - MAE: {mae_cf:.4f}, RMSE: {rmse_cf:.4f}")

    # Generate recommendations using CF
    print("Generating recommendations using User-Based CF...")
    recommendations_cf = generate_recommendations(predicted_ratings_cf, users, items, train_data)

    # Evaluate recommendations using CF
    print("Evaluating recommendations using User-Based CF...")
    precision_cf, recall_cf, f_measure_cf, ndcg_cf = evaluate_recommendations(recommendations_cf, test_data)
    print(
        f"User-Based CF Recommendation Evaluation - Precision: {precision_cf:.4f}, Recall: {recall_cf:.4f}, "
        f"F-Measure: {f_measure_cf:.4f}, NDCG: {ndcg_cf:.4f}"
    )
    
    print("__________________________________________")

        # Feature-Rich Neural Matrix Factorization - Proprietary Check
    print("__________________________________________")
    print("FEATURE-RICH NMF: PROPRIETARY CHECK")

    print("Training Feature-Rich Neural Matrix Factorization Model...")
    predicted_ratings_fr_nmf = feature_rich_nmf(train_data, movies, tags, users, items)

    # Evaluate FR-NMF predictions
    print("Evaluating FR-NMF predictions...")
    mae_fr_nmf, rmse_fr_nmf = evaluate(predicted_ratings_fr_nmf, user_item_matrix, mask)
    print(f"FR-NMF Evaluation - MAE: {mae_fr_nmf:.4f}, RMSE: {rmse_fr_nmf:.4f}")

    print("__________________________________________")

    # Output results from SVD recommendations
    print("Writing recommendations to file...")
    with open("recommendations.txt", "w") as f:
        f.write(
            f"SVD Recommendation Evaluation - Precision: {precision_svd:.4f}, Recall: {recall_svd:.4f}, "
            f"F-Measure: {f_measure_svd:.4f}, NDCG: {ndcg_svd:.4f}\n"
        )
        f.write("Top-10 Recommendations (SVD):\n")
        for user, items in recommendations_svd.items():
            f.write(f"User {user}: {', '.join(map(str, items))}\n")

if __name__ == "__main__":
    main()
