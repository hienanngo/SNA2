import streamlit as st
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
from ipysigma import Sigma
import streamlit.components.v1 as components
import os

# --- Load and preprocess data ---
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
    df["date"] = pd.to_datetime(df["date"])
    return df

# --- Filter and limit data ---
@st.cache_data
def filter_data(df):
    df = df[df["date"] >= pd.Timestamp("2022-01-01")]
    review_counts = df.groupby("reviewer_id")["listing_id"].nunique()
    core_reviewers = review_counts[review_counts >= 3].index
    df = df[df["reviewer_id"].isin(core_reviewers)]
    top_reviewers = review_counts.loc[core_reviewers].sort_values(ascending=False).head(100).index
    df = df[df["reviewer_id"].isin(top_reviewers)]

    listing_counts = df.groupby("listing_id")["reviewer_id"].nunique()
    top_listings = listing_counts[listing_counts >= 3].sort_values(ascending=False).head(50).index
    df = df[df["listing_id"].isin(top_listings)]
    return df

# --- Build bipartite graph ---
def build_bipartite_graph(df):
    B = nx.Graph()
    B.add_nodes_from(df["reviewer_id"], bipartite=0)
    B.add_nodes_from(df["listing_id"], bipartite=1)
    B.add_edges_from(zip(df["reviewer_id"], df["listing_id"]))
    return B

# --- Generate ipysigma HTML ---
def generate_ipysigma_html(B, path="bipartite_network.html"):
    guest_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    listing_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]

    for i, n in enumerate(guest_nodes):
        B.nodes[n]["x"] = -1
        B.nodes[n]["y"] = i
        B.nodes[n]["color"] = "green"
        B.nodes[n]["size"] = B.degree[n] * 2

    for i, n in enumerate(listing_nodes):
        B.nodes[n]["x"] = 1
        B.nodes[n]["y"] = i
        B.nodes[n]["color"] = "blue"
        B.nodes[n]["size"] = B.degree[n] * 2

    Sigma.write_html(
        B,
        path,
        fullscreen=True,
        hide_edges_on_move=True,
        label_density=0.0
    )

# --- Streamlit app ---
st.title("Airbnb Bipartite Network: Guests and Listings")

df = load_data()
df_filtered = filter_data(df)
st.write("Sample of the filtered dataset:")
st.dataframe(df_filtered.head())

B = build_bipartite_graph(df_filtered)
html_path = "bipartite_network.html"
generate_ipysigma_html(B, html_path)

# Embed the HTML file
if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=800, scrolling=True)
else:
    st.error("Failed to generate the bipartite network visualization.")
