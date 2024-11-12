import polars as pl
import numpy as np

# Hardcoded file path
file_path = "path/to/your/file.json"

def load_and_split_data(file_path, train_size=0.8, seed=42):
    # Read JSON file into a Polars DataFrame
    df = pl.read_json(file_path)
    
    # Print the head of the DataFrame
    print("Original DataFrame head:")
    print(df.head())
    
    # Print all columns in the DataFrame
    print("\nColumns in the DataFrame:")
    print(df.columns)
    
    # Get the number of rows in the DataFrame
    n_rows = df.shape[0]
    
    # Generate random indices for splitting
    np.random.seed(seed)
    train_indices = np.random.choice(n_rows, size=int(n_rows * train_size), replace=False)
    test_indices = np.array(list(set(range(n_rows)) - set(train_indices)))
    
    # Split the DataFrame
    train_df = df.take(train_indices)
    test_df = df.take(test_indices)
    
    print(f"\nTraining set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
    
    return train_df, test_df

# Call the function
train_df, test_df = load_and_split_data(file_path)

# Print heads of the split DataFrames
print("\nTraining DataFrame head:")
print(train_df.head())

print("\nTesting DataFrame head:")
print(test_df.head())