import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx

logger = logging.getLogger(__name__)

# Pre-indexed real nodes from Elliptic dataset test partition (timesteps 35-49)
PRESET_CONFIGS = {
    "Normal Grocery / Merchant Payment": {
        "description": "Licit single-hop merchant checkout. Frictionless transaction flow with known clean peer.",
        "default_beneficiary": "freshmart.groceries@icici",
        "default_amount": 1450.0,
        "base_node_idx": 168901,  # Real test node with ~0.298 risk
        "pattern_type": "SINGLE_HOP_LICIT",
        "risk_override": 0.12,
        "topology": {
            "nodes": [
                {"id": "Vardhan (Acct #104293)", "role": "sender", "type": "Legitimate Customer", "balance": 145000.0},
                {"id": "FreshMart Hypermarket", "role": "merchant", "type": "Verified Merchant", "balance": 450000.0},
                {"id": "Merchant Settlement Bank", "role": "bank", "type": "Clearing House", "balance": 12000000.0},
            ],
            "edges": [
                ("Vardhan (Acct #104293)", "FreshMart Hypermarket", 1450.0),
                ("FreshMart Hypermarket", "Merchant Settlement Bank", 1450.0),
            ],
        },
    },
    "Multi-Hop Mule Layering": {
        "description": "Mule layering scam: Funds transferred through rapid intermediate hops before attempting off-ramp.",
        "default_beneficiary": "mule.layer1.escrow@ybl",
        "default_amount": 85000.0,
        "base_node_idx": 136285,  # Real test node with ~0.55 risk
        "pattern_type": "MULTI_HOP_LAYERING",
        "risk_override": 0.68,
        "topology": {
            "nodes": [
                {"id": "Vardhan (Acct #104293)", "role": "sender", "type": "Victim / Source", "balance": 145000.0},
                {"id": "Mule Node 1 (Layering)", "role": "mule", "type": "Layer 1 Mule", "balance": 1200.0},
                {"id": "Mule Node 2 (Intermediate)", "role": "mule", "type": "Layer 2 Mule", "balance": 850.0},
                {"id": "Mule Node 3 (Splitter)", "role": "mule", "type": "Layer 3 Mule", "balance": 420.0},
                {"id": "Off-Ramp Liquidity Hub", "role": "cashout", "type": "P2P Crypto Cashout", "balance": 890000.0},
            ],
            "edges": [
                ("Vardhan (Acct #104293)", "Mule Node 1 (Layering)", 85000.0),
                ("Mule Node 1 (Layering)", "Mule Node 2 (Intermediate)", 84200.0),
                ("Mule Node 2 (Intermediate)", "Mule Node 3 (Splitter)", 83800.0),
                ("Mule Node 3 (Splitter)", "Off-Ramp Liquidity Hub", 83000.0),
            ],
        },
    },
    "Circular Wash Smurfing": {
        "description": "Circular laundering syndicate: Funds washed through cyclic intermediary accounts to obscure origin.",
        "default_beneficiary": "crypto.wash.ring.nodeA@paytm",
        "default_amount": 95000.0,
        "base_node_idx": 136439,  # Real test node with high illicit risk
        "pattern_type": "CIRCULAR_WASH_SMURFING",
        "risk_override": 0.94,
        "topology": {
            "nodes": [
                {"id": "Vardhan (Acct #104293)", "role": "sender", "type": "Compromised Account", "balance": 145000.0},
                {"id": "Syndicate Account A", "role": "syndicate", "type": "Primary Laundering Ring", "balance": 185000.0},
                {"id": "Layering Mule B", "role": "mule", "type": "Shell Wallet B", "balance": 5200.0},
                {"id": "Layering Mule C", "role": "mule", "type": "Shell Wallet C", "balance": 3100.0},
            ],
            "edges": [
                ("Vardhan (Acct #104293)", "Syndicate Account A", 95000.0),
                ("Syndicate Account A", "Layering Mule B", 94000.0),
                ("Layering Mule B", "Layering Mule C", 93500.0),
                ("Layering Mule C", "Syndicate Account A", 93000.0),  # Creates directed circular cycle!
            ],
        },
    },
    "Rapid Fan-Out Aggregation": {
        "description": "Fan-out smurfing: High-frequency splitting of funds into multiple micro-mule accounts.",
        "default_beneficiary": "fanout.splitter.hub@axis",
        "default_amount": 68000.0,
        "base_node_idx": 136279,  # Real test node with elevated hub risk
        "pattern_type": "RAPID_FAN_OUT",
        "risk_override": 0.74,
        "topology": {
            "nodes": [
                {"id": "Vardhan (Acct #104293)", "role": "sender", "type": "Source Account", "balance": 145000.0},
                {"id": "Fan-Out Splitter Hub", "role": "hub", "type": "Smurfing Aggregator Hub", "balance": 34000.0},
                {"id": "Micro-Mule Wallet 1", "role": "mule", "type": "Micro-Account 1", "balance": 1500.0},
                {"id": "Micro-Mule Wallet 2", "role": "mule", "type": "Micro-Account 2", "balance": 2100.0},
                {"id": "Micro-Mule Wallet 3", "role": "mule", "type": "Micro-Account 3", "balance": 980.0},
                {"id": "Consolidated Cashout Node", "role": "cashout", "type": "Dark Liquidity Pool", "balance": 540000.0},
            ],
            "edges": [
                ("Vardhan (Acct #104293)", "Fan-Out Splitter Hub", 68000.0),
                ("Fan-Out Splitter Hub", "Micro-Mule Wallet 1", 22500.0),
                ("Fan-Out Splitter Hub", "Micro-Mule Wallet 2", 23000.0),
                ("Fan-Out Splitter Hub", "Micro-Mule Wallet 3", 22000.0),
                ("Micro-Mule Wallet 1", "Consolidated Cashout Node", 22000.0),
                ("Micro-Mule Wallet 2", "Consolidated Cashout Node", 22500.0),
            ],
        },
    },
}


def analyze_topology_cycles_and_depth(edges: List[Tuple[str, str, float]]) -> Dict[str, Any]:
    """
    Analyzes a directed network for cycles and multi-hop layering depth.

    Args:
        edges: List of (source, target, amount) tuples.

    Returns:
        Dict containing cycle detection flag, cycles found, max path depth, and warning message.
    """
    G = nx.DiGraph()
    for src, dst, _ in edges:
        G.add_edge(src, dst)

    cycles = list(nx.simple_cycles(G))
    has_cycle = len(cycles) > 0

    # Calculate longest path from sender (if acyclic) or in general
    max_depth = 1
    if not has_cycle and nx.is_directed_acyclic_graph(G):
        try:
            max_depth = nx.dag_longest_path_length(G)
        except Exception:
            max_depth = 2
    else:
        max_depth = 3

    cycle_desc = ""
    if has_cycle:
        c_str = " -> ".join(cycles[0] + [cycles[0][0]])
        cycle_desc = f"Circular Laundering Wash Loop Detected: {c_str}"

    return {
        "has_cycle": has_cycle,
        "cycles": cycles,
        "cycle_description": cycle_desc,
        "max_depth": max_depth,
        "is_multihop": max_depth >= 3,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }
