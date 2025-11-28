import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import math

# 1 - Load the iris dataset
print("1 - Loading iris_dataset.csv...")
iris_df = pd.read_csv(r'E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW05\HW_05\iris_dataset.csv')


# 2 - Define input features (X) and output label (Y)
print("\n2 - Defining features and labels...")
X = iris_df.iloc[:, :4]  # First 4 features
Y = iris_df.iloc[:, 4]   # Output label

# Get unique classes and map them properly
classes = Y.unique()
print(f"Unique classes in data: {classes}")

# Create mapping based on the actual class values in your data
if set(classes) == {0, 1, 2}:
    # If classes are numeric (0, 1, 2)
    class_name_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    class_color_map = {0: 'blue', 1: 'black', 2: 'red'}
    class_letter_map = {0: 'A', 1: 'B', 2: 'C'}
else:
    # If classes are string names
    class_name_map = {'setosa': 'setosa', 'versicolor': 'versicolor', 'virginica': 'virginica'}
    class_color_map = {'setosa': 'blue', 'versicolor': 'black', 'virginica': 'red'}
    class_letter_map = {'setosa': 'A', 'versicolor': 'B', 'virginica': 'C'}

print(f"Classes: {classes}")
print(f"Class-Color mapping: {class_color_map}")
print(f"Class-Letter mapping: {class_letter_map}")

# 3 - Calculate statistics for each feature in each class
print("\n3 - Calculating statistics for each feature in each class...")

feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

stats_results = {}

for class_val in classes:
    class_data = iris_df[iris_df.iloc[:, 4] == class_val]
    class_stats = {}
    
    for i, feature in enumerate(feature_names):
        feature_data = class_data.iloc[:, i]
        class_stats[feature] = {
            'mean': np.mean(feature_data),
            'variance': np.var(feature_data),
            'std': np.std(feature_data),
            'min': np.min(feature_data),
            'max': np.max(feature_data),
            'range': np.max(feature_data) - np.min(feature_data)
        }
    
    stats_results[class_val] = class_stats

# Display statistics
for class_val in classes:
    class_name = class_name_map.get(class_val, class_val)
    print(f"\n--- Statistics for {class_name} (Class {class_val}) ---")
    for feature in feature_names:
        stats = stats_results[class_val][feature]
        print(f"{feature}:")
        print(f"  Mean: {stats['mean']:.4f}, Variance: {stats['variance']:.4f}, "
              f"Range: {stats['range']:.4f}, Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")

# Probability function based on normal distribution
def calculate_probability(x, mean, variance):
    """
    Calculate P(x|Class) using normal distribution
    P(x|Class) = (1/(√(2π) × σ)) × e^((−1/2)×((x−μ)/σ)^2)
    """
    if variance == 0:
        return 1e-10  # Small value to avoid division by zero
    std = math.sqrt(variance)
    exponent = -0.5 * ((x - mean) / std) ** 2
    probability = (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(exponent)
    return probability

# 4 - Classification using only petal length (feature 3)
print("\n4 - Classification using only petal length (feature 3)...")

# Get petal length data and class information
petal_length_data = iris_df.iloc[:, 2]  # Feature 3 (petal length)
true_labels = iris_df.iloc[:, 4]

# Get mean and variance for petal length for each class
class_params = {}
for class_val in classes:
    class_data = iris_df[iris_df.iloc[:, 4] == class_val]
    petal_length_class = class_data.iloc[:, 2]
    class_params[class_val] = {
        'mean': np.mean(petal_length_class),
        'variance': np.var(petal_length_class)
    }

# Classify each sample based on petal length
predicted_labels_1f = []
probabilities_1f = []

for i, x in enumerate(petal_length_data):
    class_probs = {}
    for class_val in classes:
        mean = class_params[class_val]['mean']
        variance = class_params[class_val]['variance']
        prob = calculate_probability(x, mean, variance)
        class_probs[class_val] = prob
    
    # Assign to class with highest probability
    predicted_class = max(class_probs, key=class_probs.get)
    predicted_labels_1f.append(predicted_class)
    probabilities_1f.append(class_probs)

# Create visualization
plt.figure(figsize=(12, 6))

# Plot original data with true colors
plt.subplot(1, 2, 1)
for class_val in classes:
    class_indices = iris_df[iris_df.iloc[:, 4] == class_val].index
    class_name = class_name_map.get(class_val, class_val)
    plt.scatter(class_indices, petal_length_data.iloc[class_indices], 
                c=class_color_map[class_val], label=f'{class_name} (True)', alpha=0.7)
plt.title('True Classes (Petal Length)')
plt.xlabel('Sample Index')
plt.ylabel('Petal Length (cm)')
plt.legend()

# Plot predicted classes
plt.subplot(1, 2, 2)
for i, (true_class, pred_class) in enumerate(zip(true_labels, predicted_labels_1f)):
    color = class_color_map[pred_class]
    plt.scatter(i, petal_length_data.iloc[i], c=color, alpha=0.7)
plt.title('Predicted Classes (Using Only Petal Length)')
plt.xlabel('Sample Index')
plt.ylabel('Petal Length (cm)')

# Create custom legend for predicted plot
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_color_map[cls], 
                              markersize=8, label=f'Class {class_letter_map[cls]} ({class_name_map[cls]})') 
                  for cls in classes]
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.show()

# 5 - Calculate accuracy for single feature classification
print("\n5 - Calculating accuracy for single feature classification...")

correct_predictions = 0
total_predictions = len(true_labels)

for true, pred in zip(true_labels, predicted_labels_1f):
    if true == pred:
        correct_predictions += 1

accuracy_1f = (correct_predictions / total_predictions) * 100

print(f"Correct predictions: {correct_predictions}")
print(f"Total predictions: {total_predictions}")
print(f"Accuracy: {accuracy_1f:.2f}%")

# 6 - Test with new samples from iris_test_samples.csv
print("\n6 - Testing with iris_test_samples.csv...")

try:
    # Load test data
    test_df = pd.read_csv(r'E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW05\HW_05\iris_test_samples.csv')
    X_test = test_df.iloc[:, :4]
    Y_test_true = test_df.iloc[:, 4]

    # Classify test samples using only petal length
    test_petal_length = test_df.iloc[:, 2]
    test_predictions_1f = []

    for x in test_petal_length:
        class_probs = {}
        for class_val in classes:
            mean = class_params[class_val]['mean']
            variance = class_params[class_val]['variance']
            prob = calculate_probability(x, mean, variance)
            class_probs[class_val] = prob
        
        predicted_class = max(class_probs, key=class_probs.get)
        test_predictions_1f.append(predicted_class)

    # 7 - Calculate test accuracy
    print("\n7 - Calculating test accuracy for single feature...")

    test_correct_1f = 0
    for true, pred in zip(Y_test_true, test_predictions_1f):
        if true == pred:
            test_correct_1f += 1

    test_accuracy_1f = (test_correct_1f / len(Y_test_true)) * 100

    print(f"Test correct predictions: {test_correct_1f}")
    print(f"Total test samples: {len(Y_test_true)}")
    print(f"Test accuracy (1 feature): {test_accuracy_1f:.2f}%")

except FileNotFoundError:
    print("Test file 'iris_test_samples.csv' not found. Skipping test evaluation.")
    test_accuracy_1f = 0

# 8 - Classification using features 3 and 4 (petal length and width)
print("\n8 - Classification using features 3 and 4 (petal length and width)...")

# Get parameters for both features
class_params_2f = {}
for class_val in classes:
    class_data = iris_df[iris_df.iloc[:, 4] == class_val]
    class_params_2f[class_val] = {
        'petal_length': {
            'mean': np.mean(class_data.iloc[:, 2]),
            'variance': np.var(class_data.iloc[:, 2])
        },
        'petal_width': {
            'mean': np.mean(class_data.iloc[:, 3]),
            'variance': np.var(class_data.iloc[:, 3])
        }
    }

# Classify training data using two features
petal_length_data = iris_df.iloc[:, 2]
petal_width_data = iris_df.iloc[:, 3]
predicted_labels_2f = []

for i in range(len(iris_df)):
    pl = petal_length_data.iloc[i]
    pw = petal_width_data.iloc[i]
    
    class_avg_probs = {}
    
    for class_val in classes:
        # Probability for petal length
        prob_pl = calculate_probability(pl, 
                                      class_params_2f[class_val]['petal_length']['mean'],
                                      class_params_2f[class_val]['petal_length']['variance'])
        
        # Probability for petal width
        prob_pw = calculate_probability(pw,
                                      class_params_2f[class_val]['petal_width']['mean'],
                                      class_params_2f[class_val]['petal_width']['variance'])
        
        # Average probability
        avg_prob = (prob_pl + prob_pw) / 2
        class_avg_probs[class_val] = avg_prob
    
    predicted_class = max(class_avg_probs, key=class_avg_probs.get)
    predicted_labels_2f.append(predicted_class)

# 9 - Calculate accuracy for two features
print("\n9 - Calculating accuracy for two features...")

correct_predictions_2f = 0
for true, pred in zip(true_labels, predicted_labels_2f):
    if true == pred:
        correct_predictions_2f += 1

accuracy_2f = (correct_predictions_2f / total_predictions) * 100

print(f"Correct predictions (2 features): {correct_predictions_2f}")
print(f"Accuracy (2 features): {accuracy_2f:.2f}%")

# Test two-feature classification on test data
if 'test_df' in locals():
    test_pl = test_df.iloc[:, 2]
    test_pw = test_df.iloc[:, 3]
    test_predictions_2f = []

    for i in range(len(test_df)):
        pl = test_pl.iloc[i]
        pw = test_pw.iloc[i]
        
        class_avg_probs = {}
        
        for class_val in classes:
            prob_pl = calculate_probability(pl, 
                                          class_params_2f[class_val]['petal_length']['mean'],
                                          class_params_2f[class_val]['petal_length']['variance'])
            
            prob_pw = calculate_probability(pw,
                                          class_params_2f[class_val]['petal_width']['mean'],
                                          class_params_2f[class_val]['petal_width']['variance'])
            
            avg_prob = (prob_pl + prob_pw) / 2
            class_avg_probs[class_val] = avg_prob
        
        predicted_class = max(class_avg_probs, key=class_avg_probs.get)
        test_predictions_2f.append(predicted_class)

    # Calculate test accuracy for two features
    test_correct_2f = 0
    for true, pred in zip(Y_test_true, test_predictions_2f):
        if true == pred:
            test_correct_2f += 1

    test_accuracy_2f = (test_correct_2f / len(Y_test_true)) * 100

    print(f"Test accuracy (2 features): {test_accuracy_2f:.2f}%")
else:
    test_accuracy_2f = 0

# 10 - Classification using all 4 features with top-2 probabilities
print("\n10 - Classification using all 4 features with top-2 probabilities...")

# Get parameters for all features
class_params_4f = {}
for class_val in classes:
    class_data = iris_df[iris_df.iloc[:, 4] == class_val]
    class_params_4f[class_val] = {}
    
    for j in range(4):
        feature_data = class_data.iloc[:, j]
        class_params_4f[class_val][f'feature_{j}'] = {
            'mean': np.mean(feature_data),
            'variance': np.var(feature_data)
        }

# Classify using all features with top-2 probabilities
predicted_labels_4f_top2 = []

for i in range(len(iris_df)):
    class_top2_avg_probs = {}
    
    for class_val in classes:
        probabilities = []
        
        for j in range(4):
            feature_value = iris_df.iloc[i, j]
            mean = class_params_4f[class_val][f'feature_{j}']['mean']
            variance = class_params_4f[class_val][f'feature_{j}']['variance']
            
            prob = calculate_probability(feature_value, mean, variance)
            probabilities.append(prob)
        
        # Get top 2 probabilities and average them
        top2_probs = sorted(probabilities, reverse=True)[:2]
        avg_top2_prob = np.mean(top2_probs)
        class_top2_avg_probs[class_val] = avg_top2_prob
    
    predicted_class = max(class_top2_avg_probs, key=class_top2_avg_probs.get)
    predicted_labels_4f_top2.append(predicted_class)

# Calculate accuracy for 4 features with top-2
correct_predictions_4f = 0
for true, pred in zip(true_labels, predicted_labels_4f_top2):
    if true == pred:
        correct_predictions_4f += 1

accuracy_4f = (correct_predictions_4f / total_predictions) * 100

print(f"Correct predictions (4 features, top-2): {correct_predictions_4f}")
print(f"Accuracy (4 features, top-2): {accuracy_4f:.2f}%")

# Test 4-feature classification on test data
if 'test_df' in locals():
    test_predictions_4f_top2 = []

    for i in range(len(test_df)):
        class_top2_avg_probs = {}
        
        for class_val in classes:
            probabilities = []
            
            for j in range(4):
                feature_value = test_df.iloc[i, j]
                mean = class_params_4f[class_val][f'feature_{j}']['mean']
                variance = class_params_4f[class_val][f'feature_{j}']['variance']
                
                prob = calculate_probability(feature_value, mean, variance)
                probabilities.append(prob)
            
            top2_probs = sorted(probabilities, reverse=True)[:2]
            avg_top2_prob = np.mean(top2_probs)
            class_top2_avg_probs[class_val] = avg_top2_prob
        
        predicted_class = max(class_top2_avg_probs, key=class_top2_avg_probs.get)
        test_predictions_4f_top2.append(predicted_class)

    # Calculate test accuracy for 4 features with top-2
    test_correct_4f = 0
    for true, pred in zip(Y_test_true, test_predictions_4f_top2):
        if true == pred:
            test_correct_4f += 1

    test_accuracy_4f = (test_correct_4f / len(Y_test_true)) * 100

    print(f"Test accuracy (4 features, top-2): {test_accuracy_4f:.2f}%")
else:
    test_accuracy_4f = 0

# Summary of results
print("\n" + "="*50)
print("SUMMARY OF RESULTS")
print("="*50)
print(f"Training Accuracy (1 feature - petal length): {accuracy_1f:.2f}%")
print(f"Training Accuracy (2 features - petal length & width): {accuracy_2f:.2f}%")
print(f"Training Accuracy (4 features, top-2): {accuracy_4f:.2f}%")
print("-" * 50)
if 'test_df' in locals():
    print(f"Test Accuracy (1 feature - petal length): {test_accuracy_1f:.2f}%")
    print(f"Test Accuracy (2 features - petal length & width): {test_accuracy_2f:.2f}%")
    print(f"Test Accuracy (4 features, top-2): {test_accuracy_4f:.2f}%")
else:
    print("Test evaluation skipped - test file not found")
    