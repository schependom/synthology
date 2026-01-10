"""
DESCRIPTION:

    Topology Comparison Script

    Calculates key graph topology metrics for generated knowledge graphs
    and compares them to standard benchmarks (implied or hardcoded).

    Metrics:
    - Average Node Degree
    - Graph Density
    - Diameter (of Largest Connected Component)
    - Clustering Coefficient (optional)

AUTHOR:

    Vincent Van Schependom
"""

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Dict

import networkx as nx

from synthology.data_structures import KnowledgeGraph

# Benchmark stats for reference (Approximate)
BENCHMARKS = {
    "FB15k-237": {
        "nodes": 14541,
        "edges": 272115,
        "avg_degree": 37.4,  # (edges * 2) / nodes? Or out-degree?
        # Usually avg out-degree is ~18.7
        "density": 0.0013,
    },
    "WN18RR": {
        "nodes": 40943,
        "edges": 86835,
        "avg_degree": 4.2,
        "density": 0.00005,
    },
}


def calculate_metrics(kg: KnowledgeGraph) -> Dict[str, Any]:
    """Calculate topology metrics for a KnowledgeGraph."""

    # Build NetworkX graph (undirected for diameter/clustering usually, but let's check directed too)
    # We use Undirected for diameter/LCC usually
    G = nx.Graph()

    # Add nodes
    for ind in kg.individuals:
        G.add_node(ind.name)

    # Add unique edges (ignoring parallel rels for topology shape)
    edges = set()
    for t in kg.triples:
        if t.positive:
            u, v = t.subject.name, t.object.name
            if u != v:  # Ignore self-loops for some metrics
                if u > v:
                    u, v = v, u
                edges.add((u, v))

    for u, v in edges:
        G.add_edge(u, v)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {"error": "Empty graph"}

    # Degree
    degrees = [d for n, d in G.degree()]
    avg_degree = statistics.mean(degrees) if degrees else 0

    # Density
    # density = 2*E / (N*(N-1))
    density = nx.density(G)

    # LCC and Diameter
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len) if components else set()
    lcc_subgraph = G.subgraph(largest_cc)

    lcc_fraction = len(largest_cc) / num_nodes if num_nodes > 0 else 0

    try:
        # Warning: Diameter can be slow on very large graphs
        if len(largest_cc) > 0:
            diameter = nx.diameter(lcc_subgraph)
        else:
            diameter = 0
    except Exception:
        diameter = -1  # Error or disconnected

    # Clustering Coeff (Avg)
    # Global clustering coefficient
    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0

    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "avg_degree": avg_degree,
        "density": density,
        "lcc_fraction": lcc_fraction,
        "diameter": diameter,
        "avg_clustering": avg_clustering,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze KG Topology")
    parser.add_argument("--data-dir", type=str, default="data/out/train", help="Directory with CSV samples")
    parser.add_argument("--limit", type=int, default=10, help="Max samples to analyze")
    args = parser.parse_args()

    input_path = Path(args.data_dir)
    csv_files = sorted(input_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {args.data_dir}")
        sys.exit(1)

    csv_files = csv_files[: args.limit]

    print(f"Analyzing {len(csv_files)} samples from {args.data_dir}...")

    all_metrics = []

    for i, file_path in enumerate(csv_files):
        print(f"[{i + 1}/{len(csv_files)}] Loading {file_path.name}...")
        try:
            kg = KnowledgeGraph.from_csv(str(file_path))
            metrics = calculate_metrics(kg)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")

    # Aggregation
    if not all_metrics:
        print("No valid metrics computed.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("TOPOLOGY ANALYSIS REPORT")
    print("=" * 60)

    metric_keys = ["nodes", "edges", "avg_degree", "density", "lcc_fraction", "diameter", "avg_clustering"]

    print(f"{'Metric':<20} | {'Mean':<10} | {'Std Dev':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 70)

    avg_results = {}

    for key in metric_keys:
        values = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float)) and m.get(key) != -1]
        if not values:
            continue

        mean_val = statistics.mean(values)
        stdev_val = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values)
        max_val = max(values)

        avg_results[key] = mean_val

        print(f"{key:<20} | {mean_val:<10.4f} | {stdev_val:<10.4f} | {min_val:<10.4f} | {max_val:<10.4f}")

    print("-" * 70)

    # Comparison
    print("\nBENCHMARK COMPARISON (Approximate)")
    print(f"{'Dataset':<15} | {'Avg Degree':<12} | {'Density':<10}")
    print("-" * 45)

    print(
        f"{'Generated (Avg)':<15} | {avg_results.get('avg_degree', 0):<12.2f} | {avg_results.get('density', 0):<10.6f}"
    )

    for name, stats in BENCHMARKS.items():
        print(f"{name:<15} | {stats['avg_degree']:<12.2f} | {stats['density']:<10.6f}")


if __name__ == "__main__":
    main()
