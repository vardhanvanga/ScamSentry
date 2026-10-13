import networkx as nx
from pyvis.network import Network
from typing import List, Tuple, Dict, Any, Optional


def render_subgraph(
    center_node: str,
    nodes: List[Dict[str, Any]],
    edges: List[Tuple[str, str]],
    height: str = "500px",
    width: str = "100%",
    notebook: bool = False,
) -> str:
    """
    Renders an interactive Pyvis/NetworkX subgraph around a target transaction node.

    Args:
        center_node: The transaction ID being analyzed.
        nodes: List of dicts with keys 'id', 'label', 'class', 'risk_score'.
        edges: List of (source, target) tuples.
        height: HTML canvas height.
        width: HTML canvas width.
        notebook: Whether rendering inside a Jupyter notebook.

    Returns:
        HTML string of the interactive Pyvis graph visualization.
    """
    net = Network(height=height, width=width, directed=True, bgcolor="#0e1117", font_color="white")
    net.force_atlas_2based()

    # Color scheme for fraud audit
    color_map = {
        1: "#ff4b4b",   # Illicit / Fraudulent (Red)
        0: "#00c853",   # Licit / Normal (Green)
        -1: "#9e9e9e",  # Background / Unknown (Grey)
    }

    for n in nodes:
        nid = str(n["id"])
        n_class = n.get("class", -1)
        risk = n.get("risk_score", 0.0)
        color = color_map.get(n_class, "#9e9e9e")
        size = 25 if nid == str(center_node) else 15

        title = f"TxID: {nid}<br>Class: {n_class}<br>Risk: {risk:.2f}"
        net.add_node(
            nid,
            label=f"{nid[:6]}..." if len(nid) > 8 else nid,
            title=title,
            color=color,
            size=size,
            borderWidth=3 if nid == str(center_node) else 1,
        )

    for src, dst in edges:
        net.add_edge(str(src), str(dst), color="#4a4a4a")

    return net.generate_html()
