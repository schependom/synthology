"""
DESCRIPTION:
    Visualizes complete Knowledge Graphs (Train/Test samples).
    Highlights the difference between Base Facts and Inferred Facts.
    Displays Class Memberships inside nodes and highlights Negative Facts.
"""

import os

try:
    import graphviz
except ImportError:
    graphviz = None
from collections import defaultdict
from typing import Optional

from synthology.data_structures import KnowledgeGraph


class GraphVisualizer:
    def __init__(self, output_dir: str = "output_graphs"):
        self.output_dir = output_dir
        print(f"Graph Visualizations will be saved to: {self.output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    def visualize(self, kg: KnowledgeGraph, filename: str, title: Optional[str] = None) -> None:
        """
        Visualizes a KnowledgeGraph instance.
        """
        if graphviz is None:
            print("Warning: Graphviz not installed. Skipping visualization.")
            return

        name_no_ext = os.path.splitext(filename)[0]
        dot = graphviz.Digraph(comment=name_no_ext)

        # Layout settings
        dot.attr(layout="sfdp")
        dot.attr(overlap="prism")
        dot.attr(sep="+12")
        dot.attr(splines="curved")
        dot.attr(nodesep="0.8")

        if title:
            dot.attr(label=title, labelloc="t", fontsize="20")

        # ---------------------------------------------------------
        # 1. PRE-PROCESS CLASS MEMBERSHIPS
        # ---------------------------------------------------------
        # Group classes by individual to list them inside the node
        ind_classes = defaultdict(list)
        for mem in kg.memberships:
            if mem.is_member:
                ind_classes[mem.individual.name].append(mem.cls.name)

        # ---------------------------------------------------------
        # 2. ADD NODES (Individuals with Classes)
        # ---------------------------------------------------------
        added_nodes = set()

        for ind in kg.individuals:
            if ind.name not in added_nodes:
                # Construct HTML Label
                classes = ind_classes.get(ind.name, [])

                # Main Name
                label_html = f"<B>{ind.name}</B>"

                # Append Classes if they exist
                if classes:
                    # Sort for consistency
                    classes_str = "<BR/>".join(sorted(classes))
                    label_html += f"<BR/><FONT POINT-SIZE='10' COLOR='#6A1B9A'><I>{classes_str}</I></FONT>"

                # Wrap in HTML-like bracket
                final_label = f"<{label_html}>"

                dot.node(
                    ind.name,
                    label=final_label,
                    shape="box",
                    style="rounded,filled",
                    fillcolor="#FFF9C4",
                    fontname="Helvetica",
                )
                added_nodes.add(ind.name)

        # ---------------------------------------------------------
        # 3. ADD ATTRIBUTE VALUES (Literals)
        # ---------------------------------------------------------
        for attr in kg.attribute_triples:
            val_str = str(attr.value)
            node_id = f"lit_{val_str}"
            if node_id not in added_nodes:
                dot.node(
                    node_id,
                    label=val_str,
                    shape="box",
                    style="filled",
                    fillcolor="#F5F5F5",
                    fontname="Courier",
                )
                added_nodes.add(node_id)

            dot.edge(
                attr.subject.name,
                node_id,
                label=attr.predicate.name,
                fontsize="10",
                color="#757575",
            )

        # ---------------------------------------------------------
        # 4. ADD RELATIONAL EDGES
        # ---------------------------------------------------------
        for triple in kg.triples:
            # --- COLOR LOGIC (Polarity) ---
            if not triple.positive:
                break
                color = "#D32F2F"  # Red for negative
                fontcolor = "#D32F2F"
                label_prefix = "¬ "  # Logical NOT symbol
            else:
                label_prefix = ""
                # Positive Base = Black, Positive Inferred = Blue
                if triple.is_base_fact:
                    color = "black"
                    fontcolor = "black"
                else:
                    color = "#1565C0"  # Blue
                    fontcolor = "#1565C0"

            # --- STYLE LOGIC (Derivation) ---
            if triple.is_base_fact:
                style = "solid"
                penwidth = "2.5"  # Thick for visibility
            else:
                style = "dashed"
                penwidth = "1.0"  # Thin for inferred

            # Create Edge
            dot.edge(
                triple.subject.name,
                triple.object.name,
                label=f"{label_prefix}{triple.predicate.name}",
                color=color,
                style=style,
                penwidth=penwidth,
                fontcolor=fontcolor,
            )

        # Render
        output_path = os.path.join(self.output_dir, name_no_ext)
        try:
            dot.render(output_path, format="pdf", cleanup=True)
            print(f"Graph Saved: {output_path}.pdf")
        except Exception as e:
            print(f"GraphViz Error: {e}")
