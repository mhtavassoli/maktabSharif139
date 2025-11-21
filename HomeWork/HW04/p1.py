"""
Iris Dataset Analysis with Feature Engineering
Mini Project: Analyzing Iris dataset features and adding new features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets._base import Bunch
import os

# 1. Data Collection and Preparation
print("=" * 50)
print("1. DATA COLLECTION AND PREPARATION")
print("=" * 50)

# Function load_iris files from local Folder
def load_iris(base_path_input):
    # Path to folder containing the files
    base_path = base_path_input
    
    # Check which files exist in the directory
    files = os.listdir(base_path)
    print("Available files:", files)
    
    # Select data file (priority: iris.data then bezdekIris.data)
    data_file = None
    if 'iris.data' in files:
        data_file = f"{base_path}\\iris.data"
    elif 'bezdekIris.data' in files:
        data_file = f"{base_path}\\bezdekIris.data"
    else:
        raise FileNotFoundError("No data file found")
    
    print(f"Using file: {data_file}")
    
    # Read data from file
    data = pd.read_csv(data_file, header=None)
    
    # Feature names (matching the original sklearn dataset)
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 
                     'petal length (cm)', 'petal width (cm)']
    
    # Extract features (first 4 columns) and target (last column)
    iris_data = data.iloc[:, :4].values
    target = data.iloc[:, 4]
    
    # Convert species names to numerical values
    species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    target_numeric = target.map(species_mapping).values
    
    # Read dataset description if available
    descr = "Iris Plants Database - Loaded from local files"
    if 'iris.names' in files:
        try:
            with open(f"{base_path}\\iris.names", 'r') as f:
                descr = f.read()
        except:
            pass
    
    # Return Bunch object with same structure as original load_iris()
    return Bunch(
        data=iris_data,
        target=target_numeric,
        target_names=np.array(['setosa', 'versicolor', 'virginica']),
        feature_names=feature_names,
        DESCR=descr,
        filename=data_file
    )

# Load the Iris dataset
iris = load_iris(r"E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW04\p1")

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values analysis:")
print(df.isnull().sum())

# Since there are no missing values in the original Iris dataset, we'll demonstrate handling missing values
# by creating a sample scenario (commented out for actual analysis)
"""
# Example of handling missing values (if they existed)
if df.isnull().sum().any():
    print("Handling missing values...")
    # For numerical columns, fill with mean
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
"""

# Calculate correlation between features
print("\nFeature correlations:")
numerical_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
correlation_matrix = df[numerical_features].corr()
print(correlation_matrix)

# 2. Adding New Feature
print("\n" + "=" * 50)
print("2. ADDING NEW FEATURE: PETAL RATIO")
print("=" * 50)

# Calculate petal ratio (petal length / petal width)
df['petal_ratio'] = df['petal length (cm)'] / df['petal width (cm)']

# Handle potential division by zero (though Iris dataset doesn't have zero petal width)
df['petal_ratio'] = df['petal_ratio'].replace([np.inf, -np.inf], np.nan)
df['petal_ratio'] = df['petal_ratio'].fillna(df['petal_ratio'].mean())

print("New feature 'petal_ratio' added successfully!")
print("\nDataset with new feature:")
print(df.head())

print("\nPetal ratio statistics by species:")
print(df.groupby('species')['petal_ratio'].describe())

# 3. Data Visualization
print("\n" + "=" * 50)
print("3. DATA VISUALIZATION")
print("=" * 50)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 3.1 Heatmap for correlation visualization
plt.subplot(2, 3, 1)
# Include the new feature in correlation matrix
features_with_ratio = numerical_features + ['petal_ratio']
correlation_matrix_extended = df[features_with_ratio].corr()

sns.heatmap(correlation_matrix_extended, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap\n(including petal_ratio)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 3.2 Pairplot for feature relationships
plt.subplot(2, 3, 2)
# We'll create a custom pairplot for selected features
selected_features = ['sepal length (cm)', 'petal length (cm)', 'petal_ratio', 'species']
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', s=60)
plt.title('Sepal Length vs Petal Length', fontweight='bold')
plt.legend(title='Species')

plt.subplot(2, 3, 3)
sns.scatterplot(data=df, x='petal length (cm)', y='petal_ratio', hue='species', s=60)
plt.title('Petal Length vs Petal Ratio', fontweight='bold')
plt.legend(title='Species')

# 3.3 Boxplot for petal_ratio distribution by species
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='species', y='petal_ratio')
plt.title('Distribution of Petal Ratio by Species', fontweight='bold')
plt.xticks(rotation=45)

# 3.4 Violin plot for better distribution visualization
plt.subplot(2, 3, 5)
sns.violinplot(data=df, x='species', y='petal_ratio')
plt.title('Violin Plot: Petal Ratio by Species', fontweight='bold')
plt.xticks(rotation=45)

# 3.5 Scatter plot: Petal Length vs Petal Ratio
plt.subplot(2, 3, 6)
scatter = plt.scatter(df['petal length (cm)'], 
                     df['petal_ratio'], 
                     c=df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}), 
                     cmap='viridis', 
                     s=60, 
                     alpha=0.7)
plt.colorbar(scatter, label='Species (0: setosa, 1: versicolor, 2: virginica)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Ratio')
plt.title('Petal Length vs Petal Ratio', fontweight='bold')

plt.tight_layout()
plt.show()

# Additional specialized plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Enhanced pairplot-like visualization
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species', size='petal_ratio', ax=ax1)
ax1.set_title('Petal Dimensions with Petal Ratio Size', fontweight='bold')
ax1.legend(title='Species')

# Petal ratio distribution by species
sns.kdeplot(data=df, x='petal_ratio', hue='species', fill=True, alpha=0.3, ax=ax2)
ax2.set_title('Petal Ratio Distribution by Species', fontweight='bold')
ax2.legend(title='Species')

plt.tight_layout()
plt.show()

# 4. Final Analysis
print("\n" + "=" * 50)
print("4. FINAL ANALYSIS")
print("=" * 50)

print("Q1: How can the new feature (ratio) help distinguish between species?")
print("Answer: The petal_ratio feature provides additional discriminative power because:")
print("- Setosa species has the lowest petal ratio (compact, wider petals)")
print("- Versicolor has intermediate petal ratio values") 
print("- Virginica has the highest petal ratio (elongated, narrower petals)")
print("- This ratio captures the shape characteristics that might not be apparent from individual measurements")

print("\nQ2: Is there a relationship between ratio and petal length?")
# Calculate correlation between petal length and petal ratio
corr_ratio_length = df['petal length (cm)'].corr(df['petal_ratio'])
print(f"Answer: The correlation between petal length and petal ratio is: {corr_ratio_length:.3f}")
if abs(corr_ratio_length) > 0.7:
    print("This indicates a STRONG relationship between petal length and petal ratio.")
elif abs(corr_ratio_length) > 0.3:
    print("This indicates a MODERATE relationship between petal length and petal ratio.")
else:
    print("This indicates a WEAK relationship between petal length and petal ratio.")

print("\nQ3: Which species have the highest and lowest petal length to width ratio?")
ratio_by_species = df.groupby('species')['petal_ratio'].agg(['mean', 'std', 'min', 'max'])
print("\nPetal ratio statistics by species:")
print(ratio_by_species)

max_ratio_species = ratio_by_species['mean'].idxmax()
min_ratio_species = ratio_by_species['mean'].idxmin()

print(f"\n- Species with HIGHEST petal ratio: {max_ratio_species} (mean: {ratio_by_species.loc[max_ratio_species, 'mean']:.2f})")
print(f"- Species with LOWEST petal ratio: {min_ratio_species} (mean: {ratio_by_species.loc[min_ratio_species, 'mean']:.2f})")

# Additional insights
print("\n" + "=" * 50)
print("ADDITIONAL INSIGHTS")
print("=" * 50)

print("1. Feature Importance:")
print("   - Petal measurements are more discriminative than sepal measurements")
print("   - Petal ratio adds shape information beyond size measurements")
print("   - Combined features provide better separation between versicolor and virginica")

print("\n2. Classification Potential:")
print("   - Setosa is easily separable with any petal-related feature")
print("   - Petal ratio might help distinguish between versicolor and virginica")
print("   - The new feature could improve machine learning model performance")

# Save the enhanced dataset
df.to_csv('iris_dataset_with_petal_ratio.csv', index=False)
print(f"\nEnhanced dataset saved as 'iris_dataset_with_petal_ratio.csv'")
print("Project completed successfully!")