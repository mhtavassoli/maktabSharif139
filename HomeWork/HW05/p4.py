import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display basic information about the dataset
print("Dataset Shape:", iris_df.shape)
print("\nFirst 5 rows:")
print(iris_df.head())
print("\nSpecies distribution:")
print(iris_df['species'].value_counts())

# 1. Calculate statistics for each class
def calculate_statistics(df, species):
    species_data = df[df['species'] == species]
    features = iris.feature_names
    
    stats = {}
    for feature in features:
        feature_data = species_data[feature]
        stats[feature] = {
            'mean': np.mean(feature_data),
            'std': np.std(feature_data, ddof=1),
            'range': np.max(feature_data) - np.min(feature_data)
        }
    
    return pd.DataFrame(stats).T

print("\n" + "="*50)
print("STATISTICAL ANALYSIS FOR EACH CLASS")
print("="*50)

# Calculate and display statistics for each species
species_list = ['setosa', 'versicolor', 'virginica']
for species in species_list:
    print(f"\nStatistics for {species.upper()}:")
    stats_df = calculate_statistics(iris_df, species)
    print(stats_df.round(3))

# 2. Calculate correlation matrices for each class
print("\n" + "="*50)
print("CORRELATION MATRICES FOR EACH CLASS")
print("="*50)

correlation_matrices = {}
for species in species_list:
    species_data = iris_df[iris_df['species'] == species]
    numeric_data = species_data[iris.feature_names]
    corr_matrix = numeric_data.corr()
    correlation_matrices[species] = corr_matrix
    print(f"\nCorrelation Matrix for {species.upper()}:")
    print(corr_matrix.round(3))

# 3. Plot heatmaps for correlation matrices
print("\n" + "="*50)
print("HEATMAP VISUALIZATION")
print("="*50)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Correlation Matrix Heatmaps for Iris Dataset Classes', fontsize=16)

for i, species in enumerate(species_list):
    sns.heatmap(correlation_matrices[species], 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                ax=axes[i],
                vmin=-1, vmax=1)
    axes[i].set_title(f'{species.title()} - Correlation Matrix')

plt.tight_layout()
plt.show()

# 4. Identify pairs with highest correlation
print("\n" + "="*50)
print("HIGHEST CORRELATION PAIRS ANALYSIS")
print("="*50)

# Function to find top correlation pairs
def find_top_correlation_pairs(corr_matrix, species):
    corr_pairs = []
    features = corr_matrix.columns
    
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr_pairs.append({
                'pair': f"{features[i]} - {features[j]}",
                'correlation': abs(corr_matrix.iloc[i, j]),
                'actual_corr': corr_matrix.iloc[i, j]
            })
    
    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
    
    print(f"\nTop correlation pairs for {species}:")
    for pair in corr_pairs[:3]:
        print(f"  {pair['pair']}: {pair['actual_corr']:.3f}")
    
    return corr_pairs[0]  # Return the highest correlation pair

# Find highest correlation pairs for each species
highest_pairs = {}
for species in species_list:
    highest_pairs[species] = find_top_correlation_pairs(correlation_matrices[species], species)

# 5. Overall analysis
print("\n" + "="*50)
print("OVERALL CORRELATION ANALYSIS")
print("="*50)

# Calculate overall correlation matrix
overall_corr = iris_df[iris.feature_names].corr()
print("\nOverall Correlation Matrix (all species combined):")
print(overall_corr.round(3))

# Plot overall heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(overall_corr, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Overall Correlation Matrix - All Iris Species')
plt.tight_layout()
plt.show()

# Find overall highest correlation pairs
overall_highest = find_top_correlation_pairs(overall_corr, "All Species")
