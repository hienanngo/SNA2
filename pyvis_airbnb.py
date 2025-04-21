import pandas as pd
import networkx as nx
from pyvis.network import Network
import kagglehub
from kagglehub import KaggleDatasetAdapter

# --- Load the Airbnb reviews dataset ---
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "hinanng/washington-d-c-airbnb-reviews",
    "reviews (3).csv"
)

# Prep data
df["reviewer_id"] = df["reviewer_id"].astype(str)
df["listing_id"] = df["listing_id"].astype(str)

# --- Build bipartite graph ---
G = nx.Graph()
G.add_nodes_from(df["reviewer_id"], bipartite=0)
G.add_nodes_from(df["listing_id"], bipartite=1)
G.add_edges_from(zip(df["reviewer_id"], df["listing_id"]))

# --- Filter for active guests only ---
guests = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
active_guests = [g for g in guests if G.degree[g] >= 5][:300]

# --- Project to guest–guest network ---
G_guest = nx.bipartite.weighted_projected_graph(G, active_guests)

# --- Build PyVis network ---
net = Network(height="800px", width="100%", bgcolor="#222", font_color="white")
net.from_nx(G_guest)

# Customize physics
net.repulsion(node_distance=150, central_gravity=0.33)

# Simplify node labels to ranks (anonymized)
for i, node in enumerate(net.nodes):
    node["label"] = f"Guest #{i+1}"
    node["title"] = f"Connections: {len(G_guest[node['id']])}"

print("✅ Script finished — did it reach the export?")


try:
    output_file = "airbnb_guest_network.html"
    net.write_html(output_file)
    print(f"✅ Network saved to: {output_file}")
except Exception as e:
    print("❌ Error saving HTML:", e)

print("Script executed successfully.")