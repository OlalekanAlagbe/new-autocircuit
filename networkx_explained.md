---
layout: post
title: "How NetworkX Represents Attribution Graphs"
date: 2026-03-01
description: "A detailed walkthrough of how the autocircuit tooling uses NetworkX DiGraphs to store, query, and compare Neuronpedia attribution graphs."
---

# How NetworkX Represents Attribution Graphs

This document explains exactly how and why we use [NetworkX](https://networkx.org/) in `autocircuit_tools_new.py`, from loading a raw JSON graph to the cross-graph comparison that produces the 180-feature core circuit.

---

## 1. What Is a DiGraph?

NetworkX's `DiGraph` (directed graph) is the central data structure. A directed graph has **nodes** connected by **edges that have a direction** — an arrow pointing from one node to another.

In this project:

| Graph concept | What it maps to |
|---|---|
| **Node** | One SAE feature firing at one token position |
| **Edge** | Causal influence flowing from feature A to feature B |
| **Edge direction** | A → B means A's activation causally affects B's |
| **Edge weight** | How strongly A influences B (can be positive or negative) |

```
[L5/#5793 at token 4] ──(weight=+0.48)──► [L8/#13766 at token 4]
        "analogies"                           "analogies or comparisons"
```

---

## 2. The Node ID Convention

Every node in the graph has a string ID in the format:

```
"{layer}_{feature_index}_{ctx_idx}"
```

| Part | Meaning | Example |
|---|---|---|
| `layer` | Transformer layer (0–25) | `5` |
| `feature_index` | SAE feature number (0–16383) | `5793` |
| `ctx_idx` | Token position in the prompt | `4` |

So `"5_5793_4"` means: layer 5, feature #5793, firing at token position 4 (the word "Berlin" in "Paris is to France as Berlin is to").

The same SAE feature can fire at multiple token positions, producing **multiple nodes** with the same `layer` and `feature_index` but different `ctx_idx`. This is why `appearances` in `compare_graphs()` can exceed 5 (one per graph).

---

## 3. Node Attributes

Each node carries a dictionary of attributes set when the graph is built in `load_graph()`:

```python
G.add_node(node_id,
    feature        = node.get('feature'),       # int: SAE feature index
    layer          = node.get('layer'),          # int: transformer layer
    feature_type   = node.get('feature_type'),  # str: "cross layer transcoder" | "transcoder" | "logit"
    ctx_idx        = node.get('ctx_idx'),        # int: token position
    influence      = node.get('influence', 0.0),# float 0–1: causal contribution to output
    activation     = node.get('activation', 0.0),# float: raw firing magnitude
    is_target_logit= node.get('is_target_logit', False),  # bool: is this the output logit node?
)
```

Accessing a node's attributes:

```python
attrs = G.nodes['5_5793_4']
print(attrs['layer'])      # 5
print(attrs['feature'])    # 5793
print(attrs['influence'])  # 0.59
```

---

## 4. Graph-Level Metadata

Beyond nodes and edges, the `DiGraph` object itself stores prompt-level metadata in `G.graph` — a plain dictionary on the graph object:

```python
G.graph['slug']          = "analog_berlin"
G.graph['prompt']        = "Paris is to France as Berlin is to"
G.graph['prompt_tokens'] = ["Paris", " is", " to", " France", " as", " Berlin", " is", " to"]
G.graph['model']         = "gemma-2-2b"
```

This means every graph carries its own identity. When `compare_graphs()` processes five graphs, it reads `G.graph['slug']` to know which prompt each node came from.

---

## 5. Edges and Weights

Edges are added from the `links` array in the JSON:

```python
for link in data.get('links', []):
    G.add_edge(link['source'], link['target'], weight=link.get('weight', 0.0))
```

Each edge has one attribute: `weight`. Positive weight means the source feature activates the target feature; negative weight means it suppresses it.

**Querying edges for a specific node** (`get_edges_for_node()`):

```python
# All edges arriving at node_id (what feeds into it)
for u, v, data in G.in_edges(node_id, data=True):
    print(f"{u} → {node_id}  weight={data['weight']:.3f}")

# All edges leaving node_id (what it drives)
for u, v, data in G.out_edges(node_id, data=True):
    print(f"{node_id} → {v}  weight={data['weight']:.3f}")
```

---

## 6. Iterating All Nodes — the `data=True` Pattern

Throughout the code you'll see:

```python
for node_id, attrs in G.nodes(data=True):
    ...
```

The `data=True` flag tells NetworkX to yield `(node_id, attribute_dict)` pairs instead of just `node_id` strings. Without it you only get the IDs and have to do `G.nodes[node_id]` separately for each one.

This pattern is used in:
- `get_graph_summary()` — collecting all influence scores, listing layers present
- `get_top_nodes()` — sorting by influence to find the most important features
- `compare_graphs()` — building the registry of recurring features

---

## 7. The `compare_graphs()` Function — Step by Step

This is the core algorithm that produces the 510 / 210 / 119 feature counts. It takes a list of five `DiGraph` objects and finds which SAE features recur across them.

### Step 1: Build a registry keyed by `(layer, feature)`

```python
registry = {}

for G in graphs:
    slug = G.graph.get('slug', 'unknown')   # e.g. "analog_berlin"

    for node_id, attrs in G.nodes(data=True):
        # Skip logit nodes and embedding nodes — only count SAE features
        if attrs.get('feature_type') not in ('cross layer transcoder', 'transcoder'):
            continue

        layer   = attrs.get('layer')
        feature = attrs.get('feature')
        key = (str(layer), str(feature))    # e.g. ("5", "5793")

        if key not in registry:
            registry[key] = []

        registry[key].append({
            'graph_slug': slug,
            'node_id'   : node_id,          # includes ctx_idx
            'influence' : attrs.get('influence', 0.0),
            'activation': attrs.get('activation', 0.0),
        })
```

The registry maps each unique `(layer, feature_index)` pair to **every instance** of that feature firing, across all graphs and all token positions. One feature that fires at 4 positions in `analog_berlin` and 3 positions in `analog_rome` will have 7 entries in its list.

### Step 2: Filter by threshold

```python
for (layer, feature), occurrences in registry.items():
    if len(occurrences) < min_appearances:
        continue
    ...
```

`min_appearances=5` keeps features with 5+ total instances. Because prompts are short (8–10 tokens), a feature can fire at most ~8 times per graph. A feature reaching 5+ total instances almost always spans multiple distinct graphs — which is exactly what we want to identify.

### Step 3: Compute averages and sort

```python
avg_inf = sum(o['influence']  for o in occurrences) / len(occurrences)
avg_act = sum(o['activation'] for o in occurrences) / len(occurrences)

results.sort(key=lambda x: (x['appearances'], x['avg_influence']), reverse=True)
```

The final list is sorted: most-recurring first, ties broken by average influence. The top entries are the core circuit features.

---

## 8. What the Five Graphs Look Like in Memory

After `load_graph()` is called for each of the five analogy prompts:

```
graphs[0]  →  G for "analog_berlin"   (e.g. 312 nodes, 847 edges)
graphs[1]  →  G for "analog_rome"     (e.g. 298 nodes, 812 edges)
graphs[2]  →  G for "analog_tokyo"    (e.g. 305 nodes, 831 edges)
graphs[3]  →  G for "analog_teacher"  (e.g. 287 nodes, 793 edges)
graphs[4]  →  G for "analog_bird"     (e.g. 271 nodes, 764 edges)
```

Each graph is independent — same `DiGraph` type, same node/edge schema, different content. `compare_graphs(graphs)` treats them as a collection and builds the registry across all of them.

---

## 9. Why DiGraph and Not Something Simpler?

A plain dictionary or dataframe could store node attributes. NetworkX is used because:

1. **Edge traversal** — `G.in_edges()` and `G.out_edges()` give immediate access to a node's causal predecessors and successors without scanning the full edge list
2. **Graph metadata** — `G.graph` cleanly attaches prompt identity to the structure itself
3. **Composability** — the same `G` object is passed between `load_graph()`, `get_top_nodes()`, `get_edges_for_node()`, and `compare_graphs()` without conversion
4. **Future extensibility** — algorithms like shortest paths, subgraph extraction, and topological sort are one function call away if needed

---

## 10. Quick Reference — NetworkX Calls Used in This Project

| Call | What it does |
|---|---|
| `nx.DiGraph()` | Create an empty directed graph |
| `G.add_node(id, **attrs)` | Add a node with attribute dictionary |
| `G.add_edge(src, tgt, weight=w)` | Add a directed edge with weight |
| `G.nodes(data=True)` | Iterate `(node_id, attrs)` pairs |
| `G.number_of_nodes()` | Count of nodes |
| `G.number_of_edges()` | Count of edges |
| `G.in_edges(node_id, data=True)` | All edges pointing **into** a node |
| `G.out_edges(node_id, data=True)` | All edges pointing **out of** a node |
| `G.graph['key']` | Read/write graph-level metadata |
| `G.nodes[node_id]` | Access a specific node's attribute dict |
