import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['ProtT5', 'CLIPT5']
metrics = ['Top-1 Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score']
data = [[0.6, 0.55, 0.5, 0.52, 0.48],  # ProtT5
        [0.62, 0.58, 0.53, 0.54, 0.5]]  # CLIPT5



# Parameters for grouped bar plot
x = np.arange(len(metrics))  # Label locations
width = 0.35  # Width of bars

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, data[0], width, label='ProtT5', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, data[1], width, label='CLIPT5', color='lightgreen', edgecolor='black')

# Labels and title
ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Scores', fontsize=12)
ax.set_title('Benchmarking Results for ProtT5 and CLIPT5 with Bootstrapping', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15, fontsize=10)
ax.legend(fontsize=10)

# Adding gridlines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Annotating bars with values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text above bar
                    textcoords="offset points",
                    ha='center', va='bottom')

# Tight layout for better spacing
fig.tight_layout()
plt.show()
