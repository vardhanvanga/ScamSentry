import logging
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

ROLE_COLOR_MAP = {
    "sender": "#29b6f6",        # Cyan Blue
    "merchant": "#00e676",      # Emerald Green
    "bank": "#78909c",          # Slate Grey
    "mule": "#ff5252",          # Vibrant Red
    "syndicate": "#d50000",     # Deep Crimson Red
    "hub": "#ff9100",           # Amber Orange
    "cashout": "#ab47bc",       # Purple
}


def build_interactive_ego_graph_fig(
    nodes: List[Dict[str, Any]],
    edges: List[Tuple[str, str, float]],
    target_node_name: Optional[str] = None,
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Renders an interactive 2-hop directed network graph in Plotly showing
    transaction flows, node roles, balances, and cycle alerts.

    Args:
        nodes: List of node dicts with 'id', 'role', 'type', 'balance'.
        edges: List of (source_id, target_id, amount) tuples.
        target_node_name: Optional ID of the central transaction node.

    Returns:
        Tuple of (Plotly Figure, Cycle/Multi-Hop Analysis Dict).
    """
    G = nx.DiGraph()
    node_meta = {n["id"]: n for n in nodes}

    for n in nodes:
        G.add_node(n["id"], **n)

    for src, dst, amt in edges:
        G.add_edge(src, dst, amount=amt)

    # Detect cycles and multi-hop paths
    cycles = list(nx.simple_cycles(G))
    has_cycle = len(cycles) > 0

    # Layout calculation
    pos = nx.spring_layout(G, k=1.4, iterations=50, seed=42)

    # Create figure
    fig = go.Figure()

    # Draw directed arrow edges
    for src, dst, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        amt = data.get("amount", 0.0)

        # Highlight edge if part of circular cycle
        is_cycle_edge = False
        if has_cycle:
            for c in cycles:
                c_edges = list(zip(c, c[1:] + [c[0]]))
                if (src, dst) in c_edges:
                    is_cycle_edge = True
                    break

        arrow_color = "#ff1744" if is_cycle_edge else ("#ffa726" if src == target_node_name else "#90caf9")
        arrow_width = 3.0 if is_cycle_edge else 1.8

        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=f"₹{amt:,.0f}" if amt > 0 else "",
            font=dict(color="#e0e0e0", size=10),
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=arrow_width,
            arrowcolor=arrow_color,
            opacity=0.9,
        )

    # Prepare node scatter points
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_symbols = []
    node_texts = []
    custom_data = []

    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)

        meta = node_meta.get(node_id, {})
        role = meta.get("role", "mule")
        node_type = meta.get("type", "Unknown")
        balance = meta.get("balance", 0.0)
        degree = G.degree(node_id)

        color = ROLE_COLOR_MAP.get(role, "#ff5252")
        size = 28 if node_id == target_node_name else 22
        symbol = "circle"

        node_colors.append(color)
        node_sizes.append(size)
        node_symbols.append(symbol)
        node_texts.append(f"<b>{node_id}</b>")
        custom_data.append([node_id, node_type, balance, degree, role.upper()])

    # Node Scatter Trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_texts,
        textposition="top center",
        textfont=dict(color="#ffffff", size=11),
        marker=dict(
            color=node_colors,
            size=node_sizes,
            symbol=node_symbols,
            line=dict(width=2, color="#ffffff"),
            opacity=0.95,
        ),
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Type: %{customdata[1]}<br>"
            "Account Balance: ₹%{customdata[2]:,.2f}<br>"
            "Degree Centrality: %{customdata[3]}<br>"
            "Network Role: %{customdata[4]}<extra></extra>"
        ),
    )

    fig.add_trace(node_trace)

    # Layout styling for dark-mode banking switch
    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=450,
    )

    cycle_info = {
        "has_cycle": has_cycle,
        "cycles": cycles,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }

    return fig, cycle_info


def create_shap_horizontal_bar(top_features: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders a clean horizontal bar chart delineating Tabular attributes vs.
    Graph-Temporal Transformer Structural Embeddings.
    """
    if not top_features:
        # Placeholder baseline
        top_features = [
            {"feature": "raw_feat_1", "domain": "Tabular", "shap_delta": 0.05, "feature_value": 0.12},
            {"feature": "raw_feat_40", "domain": "Tabular", "shap_delta": -0.02, "feature_value": -0.23},
        ]

    # Invert so highest feature is at top of horizontal chart
    features_rev = top_features[::-1]
    y_labels = [item["feature"] for item in features_rev]
    x_vals = [item["shap_delta"] for item in features_rev]
    domains = [item["domain"] for item in features_rev]
    vals = [item.get("feature_value", 0.0) for item in features_rev]

    # Differentiate colors by domain
    bar_colors = [
        "#ff5252" if dom == "GNN_Embedding" or "graph_emb" in y_labels[i] else "#42a5f5"
        for i, dom in enumerate(domains)
    ]

    fig = go.Figure(
        go.Bar(
            x=x_vals,
            y=y_labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="#ffffff", width=1)),
            text=[f"{v:+.3f} (Val: {vals[i]:.2f})" for i, v in enumerate(x_vals)],
            textposition="auto",
            hovertemplate=(
                "<b>Feature: %{y}</b><br>"
                "SHAP Impact Delta: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title=dict(
            text="Local SHAP Feature Attribution (Red = GNN Structural, Blue = Tabular)",
            font=dict(color="#ffffff", size=13),
        ),
        xaxis=dict(
            title=dict(text="SHAP Value (Push Toward Fraud)", font=dict(color="#b0bec5")),
            showgrid=True,
            gridcolor="#263238",
            zeroline=True,
            zerolinecolor="#78909c",
            tickfont=dict(color="#b0bec5"),
        ),
        yaxis=dict(
            tickfont=dict(color="#ffffff", size=11),
        ),
        margin=dict(l=10, r=20, t=40, b=30),
        height=280,
    )

    return fig
