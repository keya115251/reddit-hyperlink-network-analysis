# =====================================
# IMPORTS
# =====================================
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# =====================================
# BUILD GRAPH
# =====================================
def build_graph(df):
    print("[INFO] Building directed graph...")

    G = nx.from_pandas_edgelist(
        df,
        source='SOURCE_SUBREDDIT',
        target='TARGET_SUBREDDIT',
        create_using=nx.DiGraph()
    )

    print(f"[INFO] Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


# =====================================
# CENTRALITY MEASURES
# =====================================
def compute_centrality(G):
    print("[INFO] Computing centrality measures...")

    degree = nx.degree_centrality(G)
    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G, k=100)

    centrality_df = pd.DataFrame({
        'subreddit': list(degree.keys()),
        'degree': list(degree.values()),
        'pagerank': [pagerank[n] for n in degree.keys()],
        'betweenness': [betweenness[n] for n in degree.keys()]
    })

    print("[INFO] Centrality computed.")
    return centrality_df


# =====================================
# TOP N IMPORTANT NODES
# =====================================
def get_top_nodes(centrality_df, metric='degree', top_n=10):
    print(f"[INFO] Top {top_n} nodes by {metric}:")
    top_nodes = centrality_df.sort_values(by=metric, ascending=False).head(top_n)
    print(top_nodes)
    return top_nodes


# =====================================
# COMMUNITY DETECTION
# =====================================
def detect_communities(G):
    print("[INFO] Detecting communities on sampled graph...")

    from networkx.algorithms.community import greedy_modularity_communities

    # Take smaller subgraph
    sample_nodes = list(G.nodes())[:5000]
    G_small = G.subgraph(sample_nodes)

    communities = greedy_modularity_communities(G_small.to_undirected())

    print(f"[INFO] Communities found: {len(communities)}")
    return communities


# =====================================
# GRAPH VISUALIZATION
# =====================================
def visualize_graph_clean(G, centrality_df, sample_size=100):
    print("[INFO] Visualizing graph with important nodes highlighted...")

    # Sample nodes
    sample_nodes = list(G.nodes())[:sample_size]
    subgraph = G.subgraph(sample_nodes)

    # Get top important nodes (by PageRank)
    top_nodes = centrality_df.sort_values(by='pagerank', ascending=False).head(10)
    important_nodes = set(top_nodes['subreddit'])

    # Assign colors
    node_colors = [
        'red' if node in important_nodes else 'blue'
        for node in subgraph.nodes()
    ]

    # Layout
    pos = nx.spring_layout(subgraph, seed=42)

    # Draw
    plt.figure(figsize=(10,8))
    nx.draw(
        subgraph,
        pos,
        node_color=node_colors,
        node_size=80,
        with_labels=False
    )

    plt.title("Subreddit Network (Important Nodes Highlighted)")
    plt.show()

# =====================================
# VISUALIZE COMMUNITIES
# =====================================
def visualize_communities_clean(G, communities):
    print("[INFO] Better community visualization...")

    sample_nodes = list(G.nodes())[:100]
    subgraph = G.subgraph(sample_nodes)

    pos = nx.spring_layout(subgraph, seed=42)

    plt.figure(figsize=(10,8))

    for i, community in enumerate(communities):
        nodes = list(community.intersection(sample_nodes))
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=nodes,
            node_size=80,
            label=f"C{i}"
        )

    nx.draw_networkx_edges(subgraph, pos, alpha=0.3)

    plt.legend()
    plt.title("Community Structure (Improved)")
    plt.show()

# =====================================
# MAIN GRAPH PIPELINE
# =====================================
def graph_analysis_pipeline(df):
    G = build_graph(df)

    centrality_df = compute_centrality(G)

    # Show top nodes
    get_top_nodes(centrality_df, 'degree')
    get_top_nodes(centrality_df, 'pagerank')

    # Community detection
    communities = detect_communities(G)

    # Visualizations
    visualize_graph_clean(G, centrality_df)
    visualize_communities_clean(G, communities)

    print("[INFO] Graph analysis complete.")

    return G, centrality_df, communities



# =====================================
# RUN
# =====================================
G, centrality_df, communities = graph_analysis_pipeline(df)