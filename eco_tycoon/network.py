from __future__ import annotations
import networkx as nx
import pandas as pd


def build_dependency_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from dependency edges."""
    required = ["source", "target", "weight"]
    missing = [c for c in required if c not in edges.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["source"], row["target"], weight=float(row["weight"]))
    return G


def dependency_scores(graph: nx.DiGraph) -> pd.DataFrame:
    """Calculate centrality scores for species in the dependency network."""
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=["species", "in_degree", "out_degree", "weighted_in_degree", "weighted_out_degree", "pagerank"]
        )

    # Degree centralities
    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())

    # Weighted degree centralities
    weighted_in = dict(graph.in_degree(weight="weight"))
    weighted_out = dict(graph.out_degree(weight="weight"))

    # PageRank
    try:
        pagerank = nx.pagerank(graph, weight="weight", max_iter=1000)
    except Exception:
        pagerank = {node: 0.0 for node in graph.nodes()}

    # Combine into DataFrame
    all_nodes = set(graph.nodes())
    records = []
    for node in all_nodes:
        records.append(
            {
                "species": node,
                "in_degree": in_deg.get(node, 0),
                "out_degree": out_deg.get(node, 0),
                "weighted_in_degree": weighted_in.get(node, 0.0),
                "weighted_out_degree": weighted_out.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
            }
        )

    df = pd.DataFrame(records)
    return df.sort_values("pagerank", ascending=False).reset_index(drop=True)