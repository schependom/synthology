"""
Analyze the connectivity structure of sampled base facts.
Test hypothesis: can we get longer chains with more relations?
"""

from collections import defaultdict

import pandas as pd

# Load the test data
targets = pd.read_csv("data/exp2/baseline/family_tree_singlepass_test/train/targets.csv")
facts = pd.read_csv("data/exp2/baseline/family_tree_singlepass_test/train/facts.csv")

# Look at just one sample to understand structure
sample_1000 = facts[facts["sample_id"] == "1000"]
print("Sample 1000 base facts:")
print(f"Total base facts: {len(sample_1000)}")

# Count relation triples (exclude rdf:type)
relation_facts = sample_1000[sample_1000["predicate"] != "rdf:type"]
print(f"Relation triples (non-type): {len(relation_facts)}")

# Get unique individuals
individuals = set(sample_1000["subject"].unique()) | set(
    sample_1000[sample_1000["predicate"] != "rdf:type"]["object"].unique()
)
print(f"Total unique individuals: {len(individuals)}")

# Analyze connectivity: build a directed graph
graph = defaultdict(list)
for _, row in relation_facts.iterrows():
    s, p, o = row["subject"], row["predicate"], row["object"]
    graph[s].append((o, p))

# Find individuals with highest degree
out_degrees = {node: len(neighbors) for node, neighbors in graph.items()}
sorted_degrees = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)

print("\nTop individuals by outgoing relations:")
for ind, deg in sorted_degrees[:5]:
    print(f"  {ind}: {deg} outgoing edges")
    for target, pred in graph[ind][:3]:
        print(f"    --{pred}--> {target}")


# Try to find any path of length >= 3 (3 hops = 4 nodes)
def dfs_paths(graph, start, max_depth=4, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = [start]

    visited.add(start)

    if len(path) >= max_depth:
        return [path]

    all_paths = []
    for next_node, pred in graph.get(start, []):
        if next_node not in visited:
            new_visited = visited.copy()
            paths = dfs_paths(graph, next_node, max_depth, new_visited, path + [next_node])
            all_paths.extend(paths)

    return all_paths


print("\nSearching for paths of length >= 3 (4 nodes = 3 hops)...")
found_count = 0
for start_node in graph.keys():
    paths = dfs_paths(graph, start_node, max_depth=4)
    found_count += len(paths)
    if paths:
        for path in paths[:1]:
            print(f"  Found: {' -> '.join(path)}")

print(f"Total 3-hop+ paths found: {found_count}")

print("\n--- Relation Statistics ---")
subject_counts = relation_facts["subject"].value_counts()
print(f"Average relations per individual: {subject_counts.mean():.2f}")
print(f"Max relations from one individual: {subject_counts.max()}")
print(f"Min relations: {subject_counts.min()}")

# Calculate what we'd need for better connectivity
print("\n--- What would be needed for multi-hop reasoning? ---")
n_rels_needed_for_3hop = len(individuals) * 3  # Rough estimate: 3 edges per node
print(f"Current relations: {len(relation_facts)}")
print(f"Estimated for 3-hop paths: ~{n_rels_needed_for_3hop} (current 3x oversampling)")
print(f"Multiplier needed: {n_rels_needed_for_3hop / len(relation_facts):.1f}x")
