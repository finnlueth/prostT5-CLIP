import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import defaultdict, Counter
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score, precision_score, recall_score,
    f1_score, log_loss, hamming_loss, classification_report, confusion_matrix
)
from sklearn.utils import resample
from datasets import Dataset, load_from_disk

########################## Based on Mariam´s code ############################################

# Define the data path
data_path = os.path.join("C:/Users/ameli/OneDrive/Dokumente/Amelie/Master/train_val_GO")
absolute_path = os.path.abspath(data_path)

# Print the absolute path
print(absolute_path)

# Load the dataset
dataset = load_from_disk(absolute_path)
print(dataset)

# Access and process the train and test splits
train_df = dataset['train'].to_pandas()[['identifier', 'GO Name']]
train_df['test_or_train'] = 'train'

test_df = dataset['test'].to_pandas()[['identifier', 'GO Name']]
test_df['test_or_train'] = 'test'

# Combine train and test DataFrames
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Reorder columns to match the specified order
combined_df = combined_df[['identifier', 'test_or_train', 'GO Name']]

# Save the resulting DataFrame to a CSV file
output_csv_path = "C:/Users/ameli/OneDrive/Dokumente/Amelie/Master/train_val_GO/combined_GO_data.csv"
combined_df.to_csv(output_csv_path, index=False)

print(f"DataFrame saved to {output_csv_path}")

import pandas as pd

# Load the uploaded dataset
file_path = 'C:/Users/ameli/OneDrive/Dokumente/Amelie/Master/train_val_GO/combined_GO_data.csv'
data = pd.read_csv(file_path)

# Display the head and tail of the dataset
data_head = data.head()
data_tail = data.tail()

print(data_head, data_tail)

file_path = "C:/Users/ameli/OneDrive/Dokumente/Amelie/Master/train_val_GO/combined_GO_data.csv"
data = pd.read_csv(file_path)
top_go_terms_from_csv = data.groupby('GO Name').size().nlargest(10).index

filtered_data_csv = data[data['GO Name'].isin(top_go_terms_from_csv)]
pivoted_data_csv = filtered_data_csv.groupby(['GO Name', 'test_or_train']).size().unstack(fill_value=0)

pivoted_data_csv.plot(kind='bar', figsize=(12, 8), color=['steelblue', 'salmon'], edgecolor='black')

plt.title("Top 10 GO Terms Split by Train and Test Sets", fontsize=16)
plt.xlabel("GO Terms", fontsize=12)
plt.ylabel("Occurrences", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title="Category", fontsize=10)
plt.tight_layout()
plt.show()

train_test_counts = data['test_or_train'].value_counts()
plt.figure(figsize=(8, 8))
train_test_counts.plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90,
    colors=['steelblue', 'salmon'],
    labels=['Train', 'Test'],
    explode=[0.05, 0]
)
plt.title("Proportion of Train vs Test Sets", fontsize=16)
plt.ylabel('')
plt.tight_layout()
plt.show()
