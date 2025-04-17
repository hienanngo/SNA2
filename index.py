# pip install kagglehub pandas

import kagglehub
from kagglehub import KaggleDatasetAdapter
import networkx as nx
from networkx.algorithms import bipartite
from ipysigma import Sigma

# Load dataset from Kaggle
file_path = "reviews (3).csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "hinanng/washington-d-c-airbnb-reviews",
    file_path
)
print(df.head())

# Convert IDs to strings for consistency
df["reviewer_id"] = df["reviewer_id"].astype(str)
df["listing_id"] = df["listing_id"].astype(str)

# Create bipartite graph
G = nx.Graph()
G.add_nodes_from(df["reviewer_id"], bipartite=0)  # Guests
G.add_nodes_from(df["listing_id"], bipartite=1)   # Listings
edges = list(zip(df["reviewer_id"], df["listing_id"]))
G.add_edges_from(edges)

# Manually identify node groups
guests = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
listings = set(G) - guests

# Node coloring for visualization
colors = {node: ("green" if node in guests else "blue") for node in G.nodes}

# Visualize bipartite graph
Sigma(G, node_color=colors, node_size=G.degree)

# Project guest-to-guest network
G_guests = bipartite.projected_graph(G, guests)
Sigma(G_guests, node_size=G_guests.degree)

# Project listing-to-listing network
G_listings = bipartite.projected_graph(G, listings)
Sigma(G_listings, node_size=G_listings.degree)


