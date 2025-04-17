import streamlit as st
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

# App title
st.title("Airbnb Social Network Analysis - Washington D.C.")

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

df = load_data()
st.write("### Preview of the dataset", df.head())

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

# Identify sets
guests = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
listings = set(G) - guests

# Choose network type
network_type = st.radio("Choose network view", ["Bipartite", "Guest-to-Guest", "Listing-to-Listing"])

# Choose sample size
sample_size = st.slider("Limit number of nodes (for performance)", min_value=50, max_value=1000, value=200)

# Filter subgraph
sub_nodes = list(guests)[:sample_size] + list(listings)[:sample_size]
G_sub = G.subgraph(sub_nodes)

# Project network
if network_type == "Bipartite":
    G_plot = G_sub
elif network_type == "Guest-to-Guest":
    G_plot = bipartite.projected_graph(G, guests)
    G_plot = G_plot.subgraph(list(G_plot.nodes)[:sample_size])
elif network_type == "Listing-to-Listing":
    G_plot = bipartite.projected_graph(G, listings)
    G_plot = G_plot.subgraph(list(G_plot.nodes)[:sample_size])

# Plot the graph
st.write("### Network Graph")
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G_plot, seed=42)
nx.draw(
    G_plot,
    pos,
    with_labels=False,
    node_size=30,
    node_color="skyblue",
    edge_color="gray",
    alpha=0.7,
    ax=ax
)
st.pyplot(fig)
