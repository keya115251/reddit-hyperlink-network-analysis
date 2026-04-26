# =====================================
# IMPORTS
# =====================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================
# COMPUTE PAIRWISE AFFINITIES (P)
# =====================================
def compute_pairwise_affinities(X, perplexity=30.0, tol=1e-5):
    print("[INFO] Computing high-dimensional affinities...")

    n = X.shape[0]
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    P = np.zeros((n, n))
    entropy_target = np.log(perplexity)

    for i in range(n):
        beta = 1.0
        beta_min, beta_max = -np.inf, np.inf

        Di = D[i, np.concatenate((np.arange(i), np.arange(i+1, n)))]

        for _ in range(50):
            P_i = np.exp(-np.clip(Di * beta, 0, 50))
            sum_P_i = np.sum(P_i) + 1e-12

            H = np.log(sum_P_i) + beta * np.sum(Di * P_i) / sum_P_i
            H_diff = H - entropy_target

            if abs(H_diff) < tol:
                break

            if H_diff > 0:
                beta_min = beta
                beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2

        P_i = np.exp(-np.clip(Di * beta, 0, 50))
        P[i, np.concatenate((np.arange(i), np.arange(i+1, n)))] = P_i / np.sum(P_i)

    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    P = P / np.sum(P)

    return P


# =====================================
# PURE SNE IMPLEMENTATION
# =====================================
def pure_sne(X, n_components=2, perplexity=30.0, n_iter=300, lr=10, momentum=0.9):
    print("[INFO] Running Pure SNE...")

    n = X.shape[0]

    # Compute P
    P = compute_pairwise_affinities(X, perplexity)
    P *= 4.0  # early exaggeration

    # Initialize embedding
    Y = np.random.randn(n, n_components) * 0.0001
    dY = np.zeros_like(Y)
    iY = np.zeros_like(Y)

    for it in range(n_iter):
        sum_Y = np.sum(np.square(Y), axis=1)
        distances = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)

        num = np.exp(-np.clip(distances, 0, 50))
        np.fill_diagonal(num, 0)

        sum_num = np.sum(num)
        if sum_num == 0:
            sum_num = 1e-12

        Q = num / sum_num
        Q = np.maximum(Q, 1e-12)

        PQ = P - Q

        for i in range(n):
            dY[i] = 4 * np.sum(PQ[i][:, None] * (Y[i] - Y), axis=0)

        # Update
        iY = momentum * iY - lr * dY
        Y += iY

        # Center
        Y -= np.mean(Y, axis=0)

        # Stop exaggeration
        if it == 100:
            P /= 4.0

        if (it + 1) % 50 == 0:
            C = np.sum(P * np.log(P / Q))
            print(f"[INFO] Iter {it+1}, KL Divergence: {C:.4f}")

    print("[INFO] SNE completed.")
    return Y


# =====================================
# VISUALIZATION
# =====================================
def visualize_sne(Y, y):
    print("[INFO] Visualizing SNE...")

    df_plot = pd.DataFrame({
        'SNE1': Y[:, 0],
        'SNE2': Y[:, 1],
        'Sentiment': y
    })

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        x='SNE1',
        y='SNE2',
        hue='Sentiment',
        data=df_plot,
        palette='coolwarm',
        alpha=0.7,
        s=40
    )

    plt.title("Pure SNE Visualization (Reddit Dataset)")
    plt.legend(title="Sentiment (-1 vs +1)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =====================================
# MAIN PIPELINE
# =====================================
def sne_pipeline(X_scaled, y, sample_size=300):
    print("\n========== SNE PIPELINE ==========")

    # Reduce size (VERY IMPORTANT)
    X_sample = X_scaled[:sample_size]
    y_sample = y[:sample_size]

    np.random.seed(42)

    Y = pure_sne(
        X_sample,
        n_components=2,
        perplexity=30,
        n_iter=300,
        lr=10
    )

    visualize_sne(Y, y_sample)

    print("[INFO] SNE pipeline complete.\n")

    return Y


# =====================================
# RUN
# =====================================
Y_sne = sne_pipeline(X_scaled, y)