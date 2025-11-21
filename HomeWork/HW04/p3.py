import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Reading data and initial preparation
# 1. Read the CSV file using pandas
df = pd.read_csv(r'E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW04\p3\reviews.csv')

# 2. Verify the existence of ReviewText column
print("Dataset columns:", df.columns.tolist())
print("Dataset shape:", df.shape)

# 3. Check for missing values in ReviewText column
print("Missing values in ReviewText:", df['ReviewText'].isnull().sum())

# Handle missing values - remove rows with NaN in ReviewText
df = df.dropna(subset=['ReviewText'])

print("Dataset shape after handling missing values:", df.shape)

# Step 2: Extracting basic text features
# 1. Calculate text length (number of characters)
df['TextLength'] = df['ReviewText'].apply(len)

# 2. Count number of words in each text
df['WordCount'] = df['ReviewText'].apply(lambda x: len(str(x).split()))

print("\nBasic text statistics:")
print(f"Average text length: {df['TextLength'].mean():.2f} characters")
print(f"Average word count: {df['WordCount'].mean():.2f} words")

# Display first few rows with new features
print("\nFirst few rows with text features:")
print(df[['ReviewText', 'TextLength', 'WordCount']].head())

# Step 3: Word frequency analysis across all data
# 1. Combine all texts into one large text
all_text = ' '.join(df['ReviewText'].astype(str))

# 2. Convert text to tokens and clean using basic string operations
# Convert to lowercase
all_text_lower = all_text.lower()

# Remove common punctuation using replace
for punctuation in [',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}']:
    all_text_lower = all_text_lower.replace(punctuation, ' ')

# Split into words and remove empty strings
words = [word for word in all_text_lower.split() if word.strip()]

# 3. Calculate word frequency using manual counting
word_freq = {}
for word in words:
    word_freq[word] = word_freq.get(word, 0) + 1

# Convert to DataFrame for easier handling
freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
freq_df = freq_df.sort_values('Frequency', ascending=False)

# 4. Display top 10 most frequent words
top_10_df = freq_df.head(10)

print("\nTop 10 most frequent words:")
print(top_10_df)

# Step 4: Visualizing word frequency
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_df, x='Word', y='Frequency', palette='viridis')
plt.title('Top 10 Most Frequent Words in Reviews', fontsize=16, fontweight='bold')
plt.xlabel('Words', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Additional visualization: Distribution of text lengths
plt.figure(figsize=(10, 6))
plt.hist(df['TextLength'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Review Text Lengths', fontsize=16, fontweight='bold')
plt.xlabel('Text Length (characters)', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Additional visualization: Distribution of word counts
plt.figure(figsize=(10, 6))
plt.hist(df['WordCount'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.title('Distribution of Word Counts in Reviews', fontsize=16, fontweight='bold')
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Step 5: Analysis and conclusions
print("\n" + "="*60)
print("ANALYSIS AND CONCLUSIONS")
print("="*60)

print("\n1. Most frequent words and their potential meaning:")
for index, row in top_10_df.iterrows():
    print(f"   '{row['Word']}': {row['Frequency']} occurrences")

print("\n2. Interpretation of frequent words:")
print("   - Common English stop words (the, and, to, etc.) are expected to appear frequently")
print("   - Product/service specific words can reveal key features customers mention")
print("   - Positive words (good, great, love, excellent) typically indicate satisfaction")
print("   - Negative words (bad, terrible, horrible, worst) usually indicate dissatisfaction")
print("   - Neutral words might represent product features or common expressions")

print("\n3. Text statistics insights:")
print(f"   - Average review length: {df['TextLength'].mean():.1f} characters")
print(f"   - Average word count: {df['WordCount'].mean():.1f} words")
print(f"   - Total reviews analyzed: {len(df)}")
print(f"   - Total unique words: {len(freq_df)}")

print("\n4. Suggested preprocessing for more accurate analysis:")
print("   - Remove common stop words (the, and, is, to, etc.)")
print("   - Handle different word forms (running, runs -> run)")
print("   - Remove numbers and special characters more thoroughly")
print("   - Consider word importance using frequency thresholds")
print("   - Analyze word correlations and co-occurrences")
print("   - Implement basic sentiment analysis based on word patterns")

# Bonus: Display basic statistics
print("\n" + "="*40)
print("BASIC STATISTICS SUMMARY")
print("="*40)
print(f"Total reviews: {len(df)}")
print(f"Total words in all reviews: {len(words)}")
print(f"Unique words: {len(freq_df)}")
print(f"Most common word: '{top_10_df.iloc[0]['Word']}' ({top_10_df.iloc[0]['Frequency']} times)")
print(f"Average words per review: {df['WordCount'].mean():.1f}")
print(f"Average characters per review: {df['TextLength'].mean():.1f}")