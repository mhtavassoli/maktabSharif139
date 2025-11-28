import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Step 1: Load the dataset
print("Step 1: Loading the dataset...")
df = pd.read_csv(r'E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW05\HW_05\iris_dataset.csv')
print(f"Dataset shape: {df.shape}")
print("Column names:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Step 2: Separate test point (sample #149) and training set
print("\nStep 2: Separating test and training data...")
test_point = df.iloc[149].copy()
train_data = df.drop(149).copy()

print(f"Test point (index 149):")
print(f"Features: {test_point[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values}")
print(f"Actual class: {test_point['target']}")

actual_class = test_point['target']
print(f"Actual class recorded: {actual_class}")

# Function to calculate Euclidean distance for multiple features
def euclidean_distance_multi(point1, point2, features):
    """Calculate Euclidean distance between two points with multiple features"""
    squared_diff = 0
    for feature in features:
        squared_diff += (point1[feature] - point2[feature])**2
    return np.sqrt(squared_diff)

# Function to find k-nearest neighbors
def find_knn(test_point, train_data, features, k=3):
    """Find k-nearest neighbors using Euclidean distance"""
    distances = []
    for idx, row in train_data.iterrows():
        dist = euclidean_distance_multi(test_point, row, features)
        distances.append((dist, row['target'], idx))
    
    # Sort by distance and return k nearest neighbors
    distances.sort(key=lambda x: x[0])
    return distances[:k]

# Function for majority voting
def majority_vote(neighbors):
    """Perform majority voting to determine predicted class"""
    class_votes = {}
    for dist, cls, idx in neighbors:
        class_votes[cls] = class_votes.get(cls, 0) + 1
    return max(class_votes, key=class_votes.get)

# Part 1: KNN with all 4 features without normalization
print("\n" + "="*60)
print("PART 1: KNN with all 4 features WITHOUT normalization")
print("="*60)

all_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
neighbors_unnormalized = find_knn(test_point, train_data, all_features, k=3)

print("Three nearest neighbors (without normalization):")
for i, (dist, cls, idx) in enumerate(neighbors_unnormalized):
    print(f"Neighbor {i+1}: Distance={dist:.4f}, Class={cls}, Index={idx}")

predicted_class_unnormalized = majority_vote(neighbors_unnormalized)
is_correct_unnormalized = predicted_class_unnormalized == actual_class

print(f"Class votes: {dict(Counter([cls for dist, cls, idx in neighbors_unnormalized]))}")
print(f"Predicted class (without normalization): {predicted_class_unnormalized}")
print(f"Actual class: {actual_class}")
print(f"Prediction correct: {is_correct_unnormalized}")

# Part 2: KNN with Z-score normalized features
print("\n" + "="*60)
print("PART 2: KNN with Z-score normalized features")
print("="*60)

# Z-score normalization function
def z_score_normalize(train_data, test_point, features):
    """Apply Z-score normalization to features"""
    train_normalized = train_data.copy()
    test_normalized = test_point.copy()
    
    for feature in features:
        # Calculate mean and std from training data only
        mean = train_data[feature].mean()
        std = train_data[feature].std()
        
        # Normalize training data
        train_normalized[feature] = (train_data[feature] - mean) / std
        
        # Normalize test point using training statistics
        test_normalized[feature] = (test_point[feature] - mean) / std
    
    return train_normalized, test_normalized

# Apply Z-score normalization
train_normalized, test_normalized = z_score_normalize(train_data, test_point, all_features)

print("Training data statistics (before normalization):")
for feature in all_features:
    print(f"{feature}: mean={train_data[feature].mean():.4f}, std={train_data[feature].std():.4f}")

print("\nNormalized test point features:")
for feature in all_features:
    print(f"{feature}: {test_normalized[feature]:.4f}")

# Find nearest neighbors with normalized features
neighbors_normalized = find_knn(test_normalized, train_normalized, all_features, k=3)

print("\nThree nearest neighbors (with normalization):")
for i, (dist, cls, idx) in enumerate(neighbors_normalized):
    print(f"Neighbor {i+1}: Distance={dist:.4f}, Class={cls}, Index={idx}")

predicted_class_normalized = majority_vote(neighbors_normalized)
is_correct_normalized = predicted_class_normalized == actual_class

print(f"Class votes: {dict(Counter([cls for dist, cls, idx in neighbors_normalized]))}")
print(f"Predicted class (with normalization): {predicted_class_normalized}")
print(f"Actual class: {actual_class}")
print(f"Prediction correct: {is_correct_normalized}")

# Part 3: Comparison of results
print("\n" + "="*60)
print("PART 3: COMPARISON OF RESULTS")
print("="*60)

print("COMPARISON SUMMARY:")
print(f"{'Method':<25} {'Predicted Class':<20} {'Actual Class':<15} {'Correct':<10}")
print(f"{'-'*70}")
print(f"{'Without Normalization':<25} {predicted_class_unnormalized:<20} {actual_class:<15} {is_correct_unnormalized:<10}")
print(f"{'With Z-score Normalization':<25} {predicted_class_normalized:<20} {actual_class:<15} {is_correct_normalized:<10}")

# Additional analysis: Show how normalization affected distances
print("\nDISTANCE COMPARISON:")
print("Neighbors without normalization:")
for i, (dist, cls, idx) in enumerate(neighbors_unnormalized):
    neighbor_features = train_data.loc[idx, all_features].values
    print(f"  Neighbor {i+1}: Distance={dist:.4f}, Features={neighbor_features}")

print("Neighbors with normalization:")
for i, (dist, cls, idx) in enumerate(neighbors_normalized):
    neighbor_features = train_normalized.loc[idx, all_features].values
    print(f"  Neighbor {i+1}: Distance={dist:.4f}, Features={neighbor_features}")

# Feature ranges analysis
print("\nFEATURE RANGE ANALYSIS (Original Data):")
for feature in all_features:
    min_val = train_data[feature].min()
    max_val = train_data[feature].max()
    range_val = max_val - min_val
    print(f"{feature}: min={min_val:.2f}, max={max_val:.2f}, range={range_val:.2f}")

print("\nFEATURE RANGE ANALYSIS (Normalized Data):")
for feature in all_features:
    min_val = train_normalized[feature].min()
    max_val = train_normalized[feature].max()
    range_val = max_val - min_val
    print(f"{feature}: min={min_val:.2f}, max={max_val:.2f}, range={range_val:.2f}")
    