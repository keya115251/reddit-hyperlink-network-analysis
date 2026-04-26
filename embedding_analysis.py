# =====================================
# IMPORTS
# =====================================
from gensim.models import Word2Vec, FastText
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# =====================================
# BUILD "SENTENCES" FROM HYPERLINK CHAINS
# Each row = [source, target], grouped by
# timestamp into sequences (like sentences)
# =====================================
def build_subreddit_sentences(df, method='pairs'):
    """
    Treat source->target pairs as 2-word 'sentences'.
    Group by source to create longer context windows.
    """
    print("[INFO] Building subreddit sentences...")

    if method == 'pairs':
        # Simple: each edge is a 2-token sentence
        sentences = [
            [row['SOURCE_SUBREDDIT'], row['TARGET_SUBREDDIT']]
            for _, row in df.iterrows()
        ]

    elif method == 'source_grouped':
        # Each source subreddit's targets form one sentence
        # Captures richer context: "askreddit talked to [a, b, c, d]"
        sentences = (
            df.groupby('SOURCE_SUBREDDIT')['TARGET_SUBREDDIT']
            .apply(list)
            .reset_index()
        )
        sentences = [
            [src] + targets
            for src, targets in zip(
                sentences['SOURCE_SUBREDDIT'],
                sentences['TARGET_SUBREDDIT']
            )
        ]

    print(f"[INFO] Built {len(sentences)} sentences.")
    return sentences

# =====================================
# TRAIN WORD2VEC
# =====================================
def train_word2vec(sentences, vector_size=64, window=5, min_count=5, epochs=10):
    print("[INFO] Training Word2Vec...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,   # ignore subreddits with < 5 appearances
        workers=4,
        epochs=epochs,
        sg=1                   # Skip-gram (better for rare words)
    )
    print(f"[INFO] Word2Vec trained. Vocabulary size: {len(model.wv)}")
    return model

# =====================================
# TRAIN FASTTEXT (bonus - handles rare/unseen subreddits)
# =====================================
def train_fasttext(sentences, vector_size=64, window=5, min_count=5, epochs=10):
    print("[INFO] Training FastText...")
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        sg=1
    )
    print(f"[INFO] FastText trained. Vocabulary size: {len(model.wv)}")
    return model

# =====================================
# BUILD EMBEDDING MATRIX WITH SENTIMENT
# Map each subreddit to its avg sentiment
# from edges where it was the source
# =====================================
def build_embedding_matrix(model, df):
    print("[INFO] Building embedding matrix...")

    # Average sentiment per source subreddit
    sentiment_map = (
        df.groupby('SOURCE_SUBREDDIT')['LINK_SENTIMENT']
        .mean()
        .to_dict()
    )

    subreddits, vectors, sentiments = [], [], []

    for sub in model.wv.index_to_key:
        if sub in sentiment_map:
            subreddits.append(sub)
            vectors.append(model.wv[sub])
            sentiments.append(sentiment_map[sub])

    X_embed = np.array(vectors)
    y_embed = np.array(sentiments)

    print(f"[INFO] Embedding matrix: {X_embed.shape}")
    return subreddits, X_embed, y_embed

# =====================================
# VISUALIZE WITH PCA
# =====================================
def visualize_embeddings_pca(X_embed, y_embed, subreddits, top_n_labels=20):
    print("[INFO] PCA on embeddings...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_embed)

    # Top subreddits to label (by embedding magnitude = most "distinctive")
    magnitudes = np.linalg.norm(X_embed, axis=1)
    top_idx = np.argsort(magnitudes)[-top_n_labels:]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y_embed, cmap='coolwarm',
        s=15, alpha=0.6
    )
    # Label top nodes
    for i in top_idx:
        plt.annotate(
            subreddits[i],
            (X_pca[i, 0], X_pca[i, 1]),
            fontsize=7, alpha=0.85
        )

    plt.colorbar(scatter, label='Avg Sentiment (−1=Neg, +1=Pos)')
    plt.title("PCA of Word2Vec Subreddit Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# =====================================
# VISUALIZE WITH t-SNE
# =====================================
def visualize_embeddings_tsne(X_embed, y_embed, subreddits, top_n_labels=20):
    print("[INFO] t-SNE on embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                init='pca', random_state=42, verbose=1)
    X_tsne = tsne.fit_transform(X_embed)

    magnitudes = np.linalg.norm(X_embed, axis=1)
    top_idx = np.argsort(magnitudes)[-top_n_labels:]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=y_embed, cmap='coolwarm',
        s=15, alpha=0.6
    )
    for i in top_idx:
        plt.annotate(
            subreddits[i],
            (X_tsne[i, 0], X_tsne[i, 1]),
            fontsize=7, alpha=0.85
        )

    plt.colorbar(scatter, label='Avg Sentiment (−1=Neg, +1=Pos)')
    plt.title("t-SNE of Word2Vec Subreddit Embeddings")
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# =====================================
# NEAREST NEIGHBOR ANALYSIS
# Answers the research question directly
# =====================================
def nearest_neighbor_analysis(model, probe_subreddits, topn=10):
    """
    For a few interesting subreddits, show their nearest
    neighbors in embedding space. If similar subreddits
    cluster together, the embeddings are meaningful.
    """
    print("\n[INFO] Nearest Neighbor Analysis")
    print("=" * 50)

    for sub in probe_subreddits:
        if sub not in model.wv:
            print(f"  '{sub}' not in vocabulary, skipping.")
            continue
        neighbors = model.wv.most_similar(sub, topn=topn)
        print(f"\nTop neighbors of r/{sub}:")
        for neighbor, score in neighbors:
            print(f"  r/{neighbor:<30} similarity: {score:.4f}")

# =====================================
# SENTIMENT PREDICTION FROM EMBEDDINGS
# Simple logistic regression to quantify
# how much embedding space encodes sentiment
# =====================================
def sentiment_prediction_from_embeddings(X_embed, y_embed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    print("\n[INFO] Testing if embeddings predict sentiment...")

    # Binarize: positive (1) vs negative (-1)
    y_binary = (y_embed >= 0).astype(int)

    clf = LogisticRegression(max_iter=500)
    scores = cross_val_score(clf, X_embed, y_binary, cv=5, scoring='accuracy')

    print(f"  5-fold CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Baseline (majority class): {max(y_binary.mean(), 1-y_binary.mean()):.4f}")

    if scores.mean() > max(y_binary.mean(), 1 - y_binary.mean()) + 0.02:
        print("  FINDING: Embeddings DO encode sentiment signal above baseline.")
    else:
        print("  FINDING: Embeddings alone have limited sentiment predictability.")

    return scores

# =====================================
# MAIN PIPELINE
# =====================================
def embedding_pipeline(df):
    print("\n========== EMBEDDING PIPELINE ==========")

    # Build sentences
    sentences = build_subreddit_sentences(df, method='source_grouped')

    # Train both models
    w2v_model = train_word2vec(sentences)
    ft_model  = train_fasttext(sentences)

    # Use Word2Vec for analysis (FastText available for comparison)
    subreddits, X_embed, y_embed = build_embedding_matrix(w2v_model, df)

    # Visualize
    visualize_embeddings_pca(X_embed, y_embed, subreddits)
    visualize_embeddings_tsne(X_embed, y_embed, subreddits)

    # Research question analysis
    probe_subs = ['askreddit', 'iama', 'politics', 'funny', 'worldnews', 'dogecoin']
    nearest_neighbor_analysis(w2v_model, probe_subs)

    # Quantify: do embeddings predict sentiment?
    scores = sentiment_prediction_from_embeddings(X_embed, y_embed)

    print("\n[INFO] Embedding pipeline complete.")
    return w2v_model, ft_model, X_embed, y_embed, subreddits

# =====================================
# RUN (df already exists from preprocessing)
# =====================================
w2v_model, ft_model, X_embed, y_embed, subreddits = embedding_pipeline(df)