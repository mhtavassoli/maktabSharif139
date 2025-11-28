import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset and check the actual column names
print("Step 1: Loading the dataset...")
df = pd.read_csv(r'E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW05\HW_05\iris_dataset.csv')
print(f"Dataset shape: {df.shape}")
print("Actual column names in dataset:")
print(df.columns.tolist())
print("\nFirst few rows of dataset:")
print(df.head())

# Step 2: Select only Sepal Length and Sepal Width features based on actual column names
print("\nStep 2: Selecting features...")
# Based on the column names you provided, we'll use these exact names
selected_features = ['sepal length (cm)', 'sepal width (cm)', 'target']
df_selected = df[selected_features].copy()
print(f"Selected features dataset shape: {df_selected.shape}")
print("Selected features data:")
print(df_selected.head())

# Step 3: Separate test point (sample #149) and training set
print("\nStep 3: Separating test and training data...")
test_point = df_selected.iloc[149].copy()  # Sample #149 (0-based indexing)
train_data = df_selected.drop(149).copy()

print(f"Test point (Ptest):")
print(f"Sepal Length: {test_point['sepal length (cm)']}, "
      f"Sepal Width: {test_point['sepal width (cm)']}, "
      f"Actual Class: {test_point['target']}")
print(f"Training data shape: {train_data.shape}")

# Store actual class for later verification
actual_class = test_point['target']
print(f"Actual class recorded: {actual_class}")

# Step 4: Find nearest neighbor with k=1 using Euclidean distance
print("\nStep 4: Finding nearest neighbor (k=1)...")

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1['sepal length (cm)'] - point2['sepal length (cm)'])**2 + 
                   (point1['sepal width (cm)'] - point2['sepal width (cm)'])**2)

# Calculate distances to all training points
distances = []
for idx, row in train_data.iterrows():
    dist = euclidean_distance(test_point, row)
    distances.append((dist, row['target'], idx))

# Sort by distance and find nearest neighbor
distances.sort(key=lambda x: x[0])
nearest_neighbor_k1 = distances[0]

print(f"Nearest neighbor distance: {nearest_neighbor_k1[0]:.4f}")
print(f"Nearest neighbor class: {nearest_neighbor_k1[1]}")
print(f"Nearest neighbor index: {nearest_neighbor_k1[2]}")

predicted_class_k1 = nearest_neighbor_k1[1]
is_correct_k1 = predicted_class_k1 == actual_class
print(f"Predicted class (k=1): {predicted_class_k1}")
print(f"Actual class: {actual_class}")
print(f"Prediction correct: {is_correct_k1}")

# Step 5: Find three nearest neighbors (k=3) and use majority voting
print("\nStep 5: Finding three nearest neighbors (k=3)...")

nearest_neighbors_k3 = distances[:3]
print("Three nearest neighbors:")
for i, (dist, cls, idx) in enumerate(nearest_neighbors_k3):
    print(f"Neighbor {i+1}: Distance={dist:.4f}, Class={cls}, Index={idx}")

# Majority voting
class_votes = {}
for dist, cls, idx in nearest_neighbors_k3:
    class_votes[cls] = class_votes.get(cls, 0) + 1

predicted_class_k3 = max(class_votes, key=class_votes.get)
is_correct_k3 = predicted_class_k3 == actual_class

print(f"\nClass votes: {class_votes}")
print(f"Predicted class (k=3): {predicted_class_k3}")
print(f"Actual class: {actual_class}")
print(f"Prediction correct: {is_correct_k3}")

# Visualization
print("\nStep 6: Creating visualization...")
plt.figure(figsize=(12, 5))

# Plot 1: All data points
plt.subplot(1, 2, 1)
# Get unique classes for coloring
unique_classes = df_selected['target'].unique()
colors = {unique_classes[0]: 'red', unique_classes[1]: 'blue', unique_classes[2]: 'green'}

for class_name, color in colors.items():
    class_data = df_selected[df_selected['target'] == class_name]
    plt.scatter(class_data['sepal length (cm)'], class_data['sepal width (cm)'], 
                c=color, label=class_name, alpha=0.7)

plt.scatter(test_point['sepal length (cm)'], test_point['sepal width (cm)'], 
           c='black', marker='*', s=200, label='Test Point', edgecolors='yellow')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('All Data Points with Test Point')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Nearest neighbors
plt.subplot(1, 2, 2)
for class_name, color in colors.items():
    class_data = train_data[train_data['target'] == class_name]
    plt.scatter(class_data['sepal length (cm)'], class_data['sepal width (cm)'], 
                c=color, label=class_name, alpha=0.3)

# Highlight test point
plt.scatter(test_point['sepal length (cm)'], test_point['sepal width (cm)'], 
           c='black', marker='*', s=200, label='Test Point', edgecolors='yellow')

# Highlight nearest neighbors
for i, (dist, cls, idx) in enumerate(nearest_neighbors_k3):
    neighbor_point = train_data.loc[idx]
    plt.scatter(neighbor_point['sepal length (cm)'], neighbor_point['sepal width (cm)'], 
               c=colors[cls], marker='s', s=100, 
               label=f'Neighbor {i+1}' if i == 0 else "", 
               edgecolors='black', linewidth=2)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Nearest Neighbors (k=3)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("FINAL RESULTS:")
print("="*50)
print(f"Actual class: {actual_class}")
print(f"Predicted class (k=1): {predicted_class_k1} - {'CORRECT' if is_correct_k1 else 'WRONG'}")
print(f"Predicted class (k=3): {predicted_class_k3} - {'CORRECT' if is_correct_k3 else 'WRONG'}")
