import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_history(log_history, train_config):
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    train_logs = log_history[log_history['loss'].notna()]
    eval_logs = log_history[log_history['eval_loss'].notna()]
    
    ax1.plot(train_logs['epoch'], train_logs['loss'], label='Training Loss', color='orange', linewidth=1)
    ax1.plot(eval_logs['epoch'], eval_logs['eval_loss'], label='Eval Loss', color='lightblue', linewidth=1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    
    mean_sim = eval_logs['eval_mean_cosine_similarity']
    std_sim = eval_logs['eval_std_cosine_similarity']
    n_samples = train_config["trainer"]["eval_sample_size"]  # Using eval sample size from config
    std_error = std_sim / np.sqrt(n_samples)
    epochs = eval_logs['epoch']
    
    ax2.plot(epochs, mean_sim, label='Mean Cosine Similarity', color='green')
    ax2.fill_between(epochs,
                     mean_sim - std_error,
                     mean_sim + std_error,
                     alpha=0.3,
                     color='green',
                     label='±1 SE')
    ax2.set_ylabel('Cosine Similarity', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.grid(True)
    plt.tight_layout()
    return fig