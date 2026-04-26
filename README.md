# Reddit Hyperlink Network Analysis

## Overview
This project analyzes a large-scale Reddit-based social network using hyperlink interactions between subreddits. It combines graph-based social network analysis, dimensionality reduction, and text embedding techniques to investigate community structure, node influence, and the factors driving link sentiment across Reddit communities.

The dataset used is the **Reddit Hyperlink Network (Body Links)** from the Stanford Network Analysis Project (SNAP), consisting of approximately 858,000 directed edges representing cross-subreddit post references.

---

## Research Question
Do subreddits with similar hyperlink behavior cluster meaningfully in embedding space, and can community identity — learned purely from link co-occurrence — predict link sentiment?

---

## Dataset
| Property | Details |
|---|---|
| Source | Stanford SNAP — Reddit Hyperlink Network |
| File used | `soc-redditHyperlinks-body.tsv` |
| Nodes | Subreddits |
| Edges | ~858,000 directed hyperlinks |
| Features | 86-dimensional property vector per edge (text, sentiment, linguistic) |
| Sentiment | +1 Positive/Neutral, −1 Negative |

---

## Requirements
```bash
pip install pandas numpy scikit-learn networkx matplotlib seaborn gensim
```

## Procedure
1. Load the Reddit hyperlink body dataset and inspect its structure.
2. Clean the dataset by removing missing values and duplicates.
3. Parse the 86-dimensional PROPERTIES feature vector from each edge.
4. Extract time-based features from timestamps (year, month, day, hour).
5. Encode source and target subreddit names as numerical labels.
6. Standardize the full feature matrix using StandardScaler.
7. Construct a directed graph of subreddit interactions using NetworkX.
8. Compute degree, PageRank, and betweenness centrality to identify influential nodes.
9. Visualize the network using a force-directed spring layout with important nodes highlighted.
10. Detect communities using greedy modularity optimization.
11. Apply PCA to project the feature matrix into two dimensions and visualize by sentiment.
12. Implement a custom SNE algorithm on a sampled subset to evaluate similarity preservation.
13. Apply t-SNE on a larger sample for improved non-linear clustering visualization.
14. Build subreddit co-occurrence sentences by grouping each source subreddit with its linked targets.
15. Train Word2Vec and FastText models on these sentences to learn subreddit embeddings.
16. Construct an embedding matrix and map each subreddit to its average link sentiment.
17. Visualize subreddit embeddings using PCA and t-SNE, annotating the most distinctive nodes.
18. Perform nearest neighbor analysis in embedding space to validate topical clustering.
19. Train a logistic regression classifier on the embeddings to test sentiment predictability.
20. Compare findings across all methods to draw conclusions about network structure and sentiment.

---

## Results Summary

### Graph Analysis
The network consists of 13,624 nodes and 40,584 edges, exhibiting a scale-free structure where a small number of subreddits dominate connectivity. r/askreddit and r/iama ranked highest in both degree and PageRank centrality. Community detection identified 113 distinct communities, reflecting Reddit's naturally fragmented, topic-driven organization.

### Dimensionality Reduction
PCA captured only 27.8% of variance across two components, producing overlapping projections with limited sentiment separability. SNE preserved local structure but was computationally expensive at scale. t-SNE produced the clearest spatial organization, confirming the presence of non-linear structure in the feature space.

### Embedding Analysis
Word2Vec embeddings trained on subreddit co-occurrence sequences successfully captured topical community identity. Nearest neighbor results showed strong semantic clustering — r/worldnews grouped with r/russia and r/ukrainianconflict, r/funny grouped with r/adviceanimals and r/reactiongifs, and r/dogecoin clustered tightly with other Dogecoin-related communities.

Logistic regression on the embeddings achieved 99.69% accuracy, identical to the majority-class baseline, due to extreme class imbalance (less than 0.31% negative links). This confirmed that embeddings do not encode sentiment.

---

## Conclusion
This study demonstrates that subreddit embeddings learned from hyperlink co-occurrence effectively capture topical community identity without any post content. However, link sentiment is not encoded in community identity — it is driven by graph-level structural factors such as degree and centrality. Subreddits with high prominence tend to receive predominantly positive or neutral links regardless of their topic. This finding highlights that sentiment in Reddit's hyperlink network is a network-level phenomenon, better modeled through graph topology than through text embeddings alone. Future work could explore graph neural networks or oversample negative links to build a more balanced sentiment classifier.

---

