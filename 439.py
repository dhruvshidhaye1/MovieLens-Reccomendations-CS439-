import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage
import polars as pl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.sparse.linalg import svds
import math
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model

# Load data with Polars
def load_data(file_path):
    return pl.read_csv(file_path)

# Weighted random sampling for train-test split
def preprocess_data(ratings, test_size=0.2):
    unique_users = ratings['userId'].unique()
    train_rows = []
    test_rows = []
    
    for user in unique_users:
        user_ratings = ratings.filter(pl.col('userId') == user)
        if user_ratings.height <= 1:  # Handle cold start for users with very few ratings
            train_rows.append(user_ratings)
            continue
        
        # Weighted random sampling
        train_indices = set(random.sample(range(user_ratings.height), int((1 - test_size) * user_ratings.height)))
        test_indices = set(range(user_ratings.height)) - train_indices
        
        train_rows.append(user_ratings[sorted(train_indices)])
        test_rows.append(user_ratings[sorted(test_indices)])
    
    train_data = pl.concat(train_rows)
    test_data = pl.concat(test_rows)
    return train_data, test_data

# Create user-item matrix with confidence weighting
def create_user_item_matrix(data, unique_users, unique_items):
    user_idx_map = {user: idx for idx, user in enumerate(unique_users)}
    item_idx_map = {item: idx for idx, item in enumerate(unique_items)}
    matrix = np.zeros((len(unique_users), len(unique_items)))

    for row in data.iter_rows(named=True):
        user_idx = user_idx_map[row['userId']]
        item_idx = item_idx_map[row['movieId']]
        rating = row['rating']
        
        # Apply confidence weighting
        confidence = 1 + math.log(1 + row['rating'])  # Higher ratings carry more confidence
        matrix[user_idx, item_idx] = rating * confidence

    return matrix

# Cold start: Use average rating for new items/users
def handle_cold_start(unique_users, unique_items, user_item_matrix):
    item_means = np.array([
        np.mean(user_item_matrix[:, i][user_item_matrix[:, i] > 0]) if np.any(user_item_matrix[:, i] > 0) else 0
        for i in range(len(unique_items))
    ])
    return item_means

# Improved SVD Recommendation
def svd_recommendation(user_item_matrix, num_factors=100):
    U, sigma, Vt = svds(user_item_matrix, k=num_factors)
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    return predicted_ratings

# Generate recommendations with personalized weighting
def generate_recommendations(predicted_ratings, unique_users, unique_items, train_data, top_n=10):
    train_items = train_data.to_dict(as_series=False)
    recommendations = {}

    for user_id in unique_users:
        user_idx = unique_users.index(user_id)
        user_predictions = predicted_ratings[user_idx]
        
        rated_items = set(train_items['movieId'][i] for i in range(len(train_items['userId'])) if train_items['userId'][i] == user_id)
        unrated_items = [i for i in range(len(unique_items)) if unique_items[i] not in rated_items]
        
        # Sort and get top N
        top_items = sorted(unrated_items, key=lambda x: -user_predictions[x])[:top_n]
        recommended_item_ids = [unique_items[i] for i in top_items]
        recommendations[user_id] = recommended_item_ids

    return recommendations

# Evaluate recommendations
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

# User-Based CF
def user_based_cf(train_data, unique_users, unique_items, user_item_matrix):
    handle_cold_start(unique_users, unique_items, user_item_matrix)
    similarity = np.corrcoef(user_item_matrix)
    predicted_ratings = np.dot(similarity, user_item_matrix) / np.array(
        [np.abs(similarity).sum(axis=1)]).T
    return predicted_ratings

# TensorFlow Neural Model
def tensorflow_model(train_data, unique_users, unique_items):
    num_users = len(unique_users)
    num_items = len(unique_items)

    # Model inputs
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, 50)(user_input)
    item_embedding = Embedding(num_items, 50)(item_input)
    user_flat = Flatten()(user_embedding)
    item_flat = Flatten()(item_embedding)
    concat = Concatenate()([user_flat, item_flat])
    dense = Dense(64, activation='relu')(concat)
    output = Dense(1)(dense)

    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.summary()

    # Train data preparation using Polars and NumPy
    user_idx_map = {user: idx for idx, user in enumerate(unique_users)}
    item_idx_map = {item: idx for idx, item in enumerate(unique_items)}

    # Use Polars `.apply()` for efficient mapping to indices
    user_indices = train_data['userId'].to_numpy()
    item_indices = train_data['movieId'].to_numpy()

    # Map users and items to indices
    user_indices = np.vectorize(user_idx_map.get)(user_indices)
    item_indices = np.vectorize(item_idx_map.get)(item_indices)
    ratings = train_data['rating'].to_numpy()

    # Train the model
    model.fit([user_indices, item_indices], ratings, epochs=5, batch_size=32, verbose=1)

    # Generate predictions
    all_user_indices = np.arange(num_users)
    all_item_indices = np.arange(num_items)
    user_grid, item_grid = np.meshgrid(all_user_indices, all_item_indices)
    user_grid_flat = user_grid.flatten()
    item_grid_flat = item_grid.flatten()
    predicted_ratings = model.predict([user_grid_flat, item_grid_flat], verbose=0)
    return predicted_ratings.reshape(num_users, num_items)

# Evaluate MAE and RMSE
def evaluate(predicted_ratings, actual_ratings, mask):
    actual = actual_ratings[mask]
    predictions = predicted_ratings[mask]
    mae = mean_absolute_error(actual, predictions)
    rmse = math.sqrt(mean_squared_error(actual, predictions))
    return mae, rmse

# Create user-item matrix with weighting and tags
def create_user_item_matrix_with_tags(data, unique_users, unique_items, tags):
    user_idx_map = {user: idx for idx, user in enumerate(unique_users)}
    item_idx_map = {item: idx for idx, item in enumerate(unique_items)}
    matrix = np.zeros((len(unique_users), len(unique_items)))

    # Create a tag dictionary for lookup
    tag_data = tags.to_dict(as_series=False)
    tag_dict = {}
    for i in range(len(tag_data['userId'])):
        user_id = tag_data['userId'][i]
        item_id = tag_data['movieId'][i]
        tag = tag_data['tag'][i]
        if user_id not in tag_dict:
            tag_dict[user_id] = {}
        if item_id not in tag_dict[user_id]:
            tag_dict[user_id][item_id] = []
        tag_dict[user_id][item_id].append(tag)

    for row in data.iter_rows(named=True):
        user_idx = user_idx_map[row['userId']]
        item_idx = item_idx_map[row['movieId']]
        rating = row['rating']

        # Apply confidence weighting based on tags
        user_tags = tag_dict.get(row['userId'], {}).get(row['movieId'], [])
        tag_score = len(user_tags)  # Number of tags indicates preference strength
        confidence = 1 + math.log(1 + rating + 0.1 * tag_score)
        matrix[user_idx, item_idx] = rating * confidence

    return matrix

# Main function
def main():
    ratings = load_data('/common/home/dhr41/Documents/ml-latest-small/ratings.csv')
    movies = load_data('/common/home/dhr41/Documents/ml-latest-small/movies.csv')
    tags = load_data('/common/home/dhr41/Documents/ml-latest-small/tags.csv')

    train_data, test_data = preprocess_data(ratings)
    unique_users = sorted(ratings['userId'].unique())
    unique_items = sorted(ratings['movieId'].unique())

    user_item_matrix = create_user_item_matrix_with_tags(train_data, unique_users, unique_items, tags)

    # SVD
    print("Running SVD...")
    predicted_ratings_svd = svd_recommendation(user_item_matrix)
    recommendations_svd = generate_recommendations(predicted_ratings_svd, unique_users, unique_items, train_data)
    precision_svd, recall_svd, f_measure_svd, ndcg_svd = evaluate_recommendations(recommendations_svd, test_data)
    mae_svd, rmse_svd = evaluate(predicted_ratings_svd, user_item_matrix, user_item_matrix > 0)
    print(f"SVD: MAE={mae_svd:.4f}, RMSE={rmse_svd:.4f}, Precision={precision_svd:.4f}, Recall={recall_svd:.4f}, F-measure={f_measure_svd:.4f}, NDCG={ndcg_svd:.4f}")
    print("WRITING SVD MODEL TOP10 Ratings to reccomendations.txt -- Models running afterwards are sanity checks.")
    # Write recommendations to file
    with open("recommendations.txt", "w") as f:
        for user, items in recommendations_svd.items():
            f.write(f"User {user}: {', '.join(map(str, items))}\n")

    # User-Based CF
    print("Running User-Based CF...")
    predicted_ratings_cf = user_based_cf(train_data, unique_users, unique_items, user_item_matrix)
    precision_cf, recall_cf, f_measure_cf, ndcg_cf = evaluate_recommendations(
        generate_recommendations(predicted_ratings_cf, unique_users, unique_items, train_data), test_data
    )
    mae_cf, rmse_cf = evaluate(predicted_ratings_cf, user_item_matrix, user_item_matrix > 0)
    print(f"User-Based CF: MAE={mae_cf:.4f}, RMSE={rmse_cf:.4f}, Precision={precision_cf:.4f}, Recall={recall_cf:.4f}, F-measure={f_measure_cf:.4f}, NDCG={ndcg_cf:.4f}")

    # TensorFlow Model
    print("Running TensorFlow Model...")
    predicted_ratings_tf = tensorflow_model(train_data, unique_users, unique_items)
    precision_tf, recall_tf, f_measure_tf, ndcg_tf = evaluate_recommendations(
        generate_recommendations(predicted_ratings_tf, unique_users, unique_items, train_data), test_data
    )
    mae_tf, rmse_tf = evaluate(predicted_ratings_tf, user_item_matrix, user_item_matrix > 0)
    print(f"TensorFlow Model: MAE={mae_tf:.4f}, RMSE={rmse_tf:.4f}, Precision={precision_tf:.4f}, Recall={recall_tf:.4f}, F-measure={f_measure_tf:.4f}, NDCG={ndcg_tf:.4f}")

    print("All models completed.")

if __name__ == "__main__":
    main()