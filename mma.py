import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import kagglehub
from kagglehub import KaggleDatasetAdapter

@st.cache_data
def load_data():
    file_path = "pro_mma_fights.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "binduvr/pro-mma-fights",
        file_path,
    )
    return df

def clean_data(df):
    # Normalize fighter names
    for col in ["fighter1_name", "fighter2_name"]:
        df[col] = df[col].str.strip().str.title()

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Generate unique fight ID
    df["fight_id"] = (
        df["fighter1_name"] + "_vs_" + df["fighter2_name"] + "_" + df["date"].dt.strftime('%Y-%m-%d')
    ).fillna("unknown")

    return df

def build_graph(df):
    G = nx.Graph()
    
    for _, row in df.iterrows():
        f1 = row["fighter1_name"]
        f2 = row["fighter2_name"]
        result = row["fighter1_result"]
        method = row["win_method"]

        # Add fighters as nodes
        G.add_node(f1)
        G.add_node(f2)

        # Edge with metadata
        G.add_edge(f1, f2, 
                   result=result,
                   method=method,
                   date=row["date"].strftime('%Y-%m-%d'),
                   event=row["event_title"])
    return G

def display_graph_pyvis(G):
    net = Network(notebook=False, height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.write_html("mma_graph.html")  # Safer than .show()
    with open("mma_graph.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=650, scrolling=True)

st.title("MMA Fight Network Explorer")

# Load and clean
df = clean_data(load_data())

# Filters (simplified for now)
df_small = df[df['organisation'] == "Ultimate Fighting Championship (UFC)"]
df_small = df_small[df_small['date'].dt.year >= 2020]  # limit size for performance

# Build graph
G = build_graph(df_small)

# Show stats
st.write(f"### Network Stats")
st.write(f"- Number of Fighters: {G.number_of_nodes()}")
st.write(f"- Number of Fights: {G.number_of_edges()}")

# Show graph
st.write("### Interactive Network Graph")
display_graph_pyvis(G)
