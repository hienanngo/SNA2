import streamlit as st
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "hinanng/washington-d-c-airbnb-reviews",
        "reviews (3).csv"
    )
    df["reviewer_id"] = df["reviewer_id"].astype(str)
    df["listing_id"] = df["listing_id"].astype(str)
    return df

st.title("🏠 Airbnb Social Network Analysis (Washington D.C.)")
df = load_data()
st.write("### 🔍 Sample of the dataset", df.head())

# Build bipartite graph
@st.cache_data
def build_graph(df):
    G = nx.Graph()
    G.add_nodes_from(df["reviewer_id"], bipartite=0)
    G.add_nodes_from(df["listing_id"], bipartite=1)
    edges = list(zip(df["reviewer_id"], df["listing_id"]))
    G.add_edges_from(edges)
    return G

G = build_graph(df)
guests = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
listings = set(G) - guests

# --- Centrality View ---
st.header("🔝 Centrality View")

# Select mode and number of nodes
mode = st.radio("Choose projection type", ["Guest-to-Guest", "Listing-to-Listing"])
limit = st.slider("Max nodes to analyze", min_value=50, max_value=2000, value=300)

if mode == "Guest-to-Guest":
    nodes = [n for n in guests if G.degree[n] > 3][:limit]
elif mode == "Listing-to-Listing":
    nodes = [n for n in listings if G.degree[n] > 3][:limit]

if len(nodes) < 2:
    st.warning("Not enough nodes to build a projection.")
else:
    G_proj = bipartite.projected_graph(G, nodes)
    st.write(f"Network: {G_proj.number_of_nodes()} nodes / {G_proj.number_of_edges()} edges")

    # Compute degree centrality
    centrality = nx.degree_centrality(G_proj)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    st.write("### 📊 Top Central Nodes")
    st.table(pd.DataFrame(top_nodes, columns=["Node ID", "Degree Centrality"]))

    # Select node to visualize
    selected_node = st.selectbox("View neighborhood of node:", [n for n, _ in top_nodes])

    # Plot ego graph
    ego = nx.ego_graph(G_proj, selected_node, radius=1)
    pos = nx.spring_layout(ego, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        ego,
        pos,
        with_labels=True,
        node_color="orange",
        edge_color="gray",
        node_size=300,
        ax=ax
    )
    plt.title(f"Ego Network of: {selected_node}")
    st.pyplot(fig)
