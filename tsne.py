# =====================================
# IMPORTS
# =====================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


# =====================================
# APPLY t-SNE
# =====================================
def apply_tsne(X, n_components=2, perplexity=30, n_iter=1000, learning_rate=200):
    print("[INFO] Running t-SNE...")

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        init='pca',
        random_state=42,
        verbose=1
    )

    Y = tsne.fit_transform(X)

    print("[INFO] t-SNE completed.")
    return Y


# =====================================
# VISUALIZATION
# =====================================
def visualize_tsne(Y, y):
    print("[INFO] Visualizing t-SNE...")

    df_plot = pd.DataFrame({
        'TSNE1': Y[:, 0],
        'TSNE2': Y[:, 1],
        'Sentiment': y
    })

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        x='TSNE1',
        y='TSNE2',
        hue='Sentiment',
        data=df_plot,
        palette='coolwarm',
        alpha=0.7,
        s=40
    )

    plt.title("t-SNE Visualization (Reddit Dataset)")
    plt.legend(title="Sentiment (-1 vs +1)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =====================================
# MAIN PIPELINE
# =====================================
def tsne_pipeline(X_scaled, y, sample_size=3000):
    print("\n========== t-SNE PIPELINE ==========")

    # Sampling (IMPORTANT for speed)
    X_sample = X_scaled[:sample_size]
    y_sample = y[:sample_size]

    Y = apply_tsne(
        X_sample,
        n_components=2,
        perplexity=30,
        n_iter=1000,
        learning_rate=200
    )

    visualize_tsne(Y, y_sample)

    print("[INFO] t-SNE pipeline complete.\n")

    return Y


# =====================================
# RUN
# =====================================
Y_tsne = tsne_pipeline(X_scaled, y)