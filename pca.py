# =====================================
# IMPORTS
# =====================================
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# =====================================
# APPLY PCA
# =====================================
def apply_pca(X_scaled, n_components=2):
    print("[INFO] Applying PCA...")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print("[INFO] PCA completed.")
    return pca, X_pca


# =====================================
# EXPLAINED VARIANCE
# =====================================
def explained_variance_analysis(pca):
    print("[INFO] Explained variance ratio:")

    variance = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance)

    for i, v in enumerate(variance):
        print(f"PC{i+1}: {v:.4f}")

    print(f"[INFO] Cumulative variance: {cumulative}")

    return variance, cumulative


# =====================================
# PLOT VARIANCE
# =====================================
def plot_variance(cumulative):
    plt.figure(figsize=(8,5))
    plt.plot(cumulative, marker='o')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Explained")
    plt.grid()
    plt.show()


# =====================================
# PCA VISUALIZATION
# =====================================
def visualize_pca(X_pca, y):
    print("[INFO] Visualizing PCA (improved)...")

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap='coolwarm',
        s=8,
        alpha=0.7
    )

    plt.title("PCA Projection of Reddit Dataset", fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Colorbar with labels
    cbar = plt.colorbar(scatter)
    cbar.set_label("Sentiment (-1 = Negative, +1 = Positive)")

    plt.grid(alpha=0.3)
    plt.show()
# =====================================
# MAIN PCA PIPELINE
# =====================================
def pca_pipeline(X_scaled, y):
    pca, X_pca = apply_pca(X_scaled)

    variance, cumulative = explained_variance_analysis(pca)

    plot_variance(cumulative)
    visualize_pca(X_pca, y)

    print("[INFO] PCA pipeline complete.")

    return pca, X_pca


# =====================================
# RUN
# =====================================
pca, X_pca = pca_pipeline(X_scaled, y)