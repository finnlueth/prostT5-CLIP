import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'C:/Users/ameli/OneDrive/Dokumente/train_metadata(reduced).tsv'
metadata_df = pd.read_csv(file_path, sep='\t')

dataset_info = {
    "Head": metadata_df.head(),
    "Info": metadata_df.info(),
    "Describe": metadata_df.describe(include='all'),
    "Null Values": metadata_df.isnull().sum(),
    "Shape": metadata_df.shape,
    "Columns": metadata_df.columns.tolist(),
}

print(dataset_info)

metadata_df['unique_go_terms'] = metadata_df['GO_terms'].apply(lambda x: len(set(x.split(','))))

plt.figure(figsize=(10, 6))
sns.histplot(metadata_df['length'], bins=50, kde=True, color='teal')
plt.title('Distribution of Protein Sequence Lengths', fontsize=14)
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(metadata_df['num_terms'], bins=50, kde=True, color='blue')
plt.title('Distribution of Number of GO Terms', fontsize=14)
plt.xlabel('Number of GO Terms', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
metadata_df['kingdom'].value_counts().plot(kind='bar', color='green')
plt.title('Kingdom Distribution', fontsize=14)
plt.xlabel('Kingdom', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
metadata_df['aspects'].value_counts().plot(kind='bar', color='purple')
plt.title('Distribution of GO Aspects', fontsize=14)
plt.xlabel('Aspects', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
top_species = metadata_df['species'].value_counts().head(10)
top_species.plot(kind='bar', color='orange')
plt.title('Top 10 Species with Most Entries', fontsize=14)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(metadata_df[['length', 'num_terms']].corr(), annot=True, cmap='coolwarm', cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(metadata_df['unique_go_terms'], bins=50, kde=True, color='cyan')
plt.title('Distribution of Unique GO Terms Per Protein', fontsize=14)
plt.xlabel('Number of Unique GO Terms', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
top_taxonomy_ids = metadata_df['taxonomyID'].value_counts().head(10)
top_taxonomy_ids.plot(kind='bar', color='darkred')
plt.title('Top 10 Most Common Taxonomy IDs', fontsize=14)
plt.xlabel('Taxonomy ID', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()
