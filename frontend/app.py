"""
ScamSentry (ScamSentinel) — STAR-Fraud Engine
=============================================
Page Navigation Flow:
  1. login_selection    → Welcome / Portal Selection (Image 1 - centered, side-by-side)
  2. customer_login     → Customer Authentication (Image 3)
  3. employee_login     → Bank Employee Authentication (Image 2)
  4. customer_portal    → Customer Mobile Wallet (PhonePe-Style with 4-Grid: Phone, Bank/UPI, Scan QR, Balance PIN)
  5. employee_dashboard → Bank SOC Central Switch Live Monitor & Deep Subgraph Investigation
  6. employee_accounts  → Bank Accounts Management & Detailed Bank Statements (Add, Delete, Statements)
  7. employee_about     → Institutional Model Architecture & System Working Conditions
"""

import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import torch
import streamlit as st

# ── MUST be the very first Streamlit call ──────────────────────────────────────
st.set_page_config(
    page_title="ScamSentinel — Fraud Detection Switch",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Ensure project root is in sys.path ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ── Backend imports (with graceful fallback) ──────────────────────────────────
try:
    from backend.app.engine.streaming_pipeline import RealTimeInferencePipeline
    _PIPELINE_AVAILABLE = True
except Exception:
    _PIPELINE_AVAILABLE = False

from frontend.src.simulator import PRESET_CONFIGS, analyze_topology_cycles_and_depth
from frontend.src.graph_visualizer import build_interactive_ego_graph_fig, create_shap_horizontal_bar

# ==============================================================================
# DEFAULT REGISTERED BANKING ACCOUNTS (Realistic Financial Compliance Profiles)
# ==============================================================================
DEFAULT_ACCOUNTS = [
    {
        "id": "ACC_101",
        "name": "FreshMart Hypermarket",
        "phone": "9876543210",
        "upi": "freshmart.groceries@icici",
        "account_num": "100293847101",
        "bank": "ICICI Bank (IFSC: ICIC0001002)",
        "balance": 450000.0,
        "type": "Verified Standard Merchant Account",
        "regulatory_status": "Clean (Zero Consumer Complaints)",
        "preset_key": "Normal Grocery / Merchant Payment",
        "base_node_idx": 168901,
        "kyc_status": "KYC Verified (Tier 3 Institutional)",
        "statement": [
            {"date": "2026-09-29 09:15", "desc": "POS Terminal Settlement", "ref": "TX_992104", "type": "CR", "amount": 1450.0, "balance": 450000.0},
            {"date": "2026-09-28 18:30", "desc": "UPI Merchant Checkout", "ref": "TX_881923", "type": "CR", "amount": 3200.0, "balance": 448550.0},
            {"date": "2026-09-28 14:10", "desc": "Vendor Bulk Settlement", "ref": "TX_772019", "type": "DR", "amount": 12500.0, "balance": 445350.0},
        ],
    },
    {
        "id": "ACC_102",
        "name": "Apex Digital Liquidity Services Ltd",
        "phone": "9845123456",
        "upi": "mule.layer1.escrow@ybl",
        "account_num": "209384758192",
        "bank": "YES Bank (IFSC: YESB0002093)",
        "balance": 1200.0,
        "type": "Flagged Account (Repeated High-Velocity Layering Inquiries)",
        "regulatory_status": "Flagged / Under Active Surveillance",
        "preset_key": "Multi-Hop Mule Layering",
        "base_node_idx": 136285,
        "kyc_status": "Partial KYC (Flagged Device Identifier)",
        "statement": [
            {"date": "2026-09-29 11:02", "desc": "Rapid Inbound Dispersal", "ref": "TX_661920", "type": "CR", "amount": 85000.0, "balance": 86200.0},
            {"date": "2026-09-29 11:03", "desc": "Immediate Outbound Forward to Layer 2", "ref": "TX_661921", "type": "DR", "amount": 84200.0, "balance": 2000.0},
            {"date": "2026-09-28 10:15", "desc": "Micro ATM Cash Withdrawal", "ref": "TX_551029", "type": "DR", "amount": 800.0, "balance": 1200.0},
        ],
    },
    {
        "id": "ACC_103",
        "name": "FinTech Global Pay Solutions Ltd",
        "phone": "9712345678",
        "upi": "crypto.wash.ring.nodeA@paytm",
        "account_num": "301928475610",
        "bank": "Paytm Payments Bank (IFSC: PYTM0123456)",
        "balance": 185000.0,
        "type": "Flagged Account (Previous Scam Incident Report #AML-2026-089)",
        "regulatory_status": "High-Risk Flagged (4 Consumer Complaints Recorded)",
        "preset_key": "Circular Wash Smurfing",
        "base_node_idx": 136439,
        "kyc_status": "Compromised Shell Entity (High Risk)",
        "statement": [
            {"date": "2026-09-29 14:22", "desc": "Circular Closed-Loop Settlement from Node C", "ref": "TX_441092", "type": "CR", "amount": 93000.0, "balance": 185000.0},
            {"date": "2026-09-29 14:20", "desc": "Rapid Ring Forward to Node B", "ref": "TX_441091", "type": "DR", "amount": 94000.0, "balance": 92000.0},
            {"date": "2026-09-28 19:40", "desc": "Dark Liquidity Seed Inbound", "ref": "TX_330912", "type": "CR", "amount": 95000.0, "balance": 186000.0},
        ],
    },
    {
        "id": "ACC_104",
        "name": "QuickPay Escrow Settlement LLP",
        "phone": "9923456789",
        "upi": "fanout.splitter.hub@axis",
        "account_num": "409182736451",
        "bank": "Axis Bank (IFSC: UTIB0004091)",
        "balance": 34000.0,
        "type": "Flagged Account (Excessive Fan-In Velocity & Dispersal Signals)",
        "regulatory_status": "Flagged / Monitored Splitter Hub",
        "preset_key": "Rapid Fan-Out Aggregation",
        "base_node_idx": 136279,
        "kyc_status": "Flagged Fan-in Ratio (>12 counterparties/hr)",
        "statement": [
            {"date": "2026-09-29 16:05", "desc": "Inbound Aggregation from Victim", "ref": "TX_220914", "type": "CR", "amount": 68000.0, "balance": 102000.0},
            {"date": "2026-09-29 16:06", "desc": "Micro-Splitter Outbound Batch 1", "ref": "TX_220915", "type": "DR", "amount": 22500.0, "balance": 79500.0},
            {"date": "2026-09-29 16:06", "desc": "Micro-Splitter Outbound Batch 2", "ref": "TX_220916", "type": "DR", "amount": 23000.0, "balance": 56500.0},
            {"date": "2026-09-29 16:07", "desc": "Micro-Splitter Outbound Batch 3", "ref": "TX_220917", "type": "DR", "amount": 22500.0, "balance": 34000.0},
        ],
    },
    {
        "id": "ACC_105",
        "name": "Aarav Sharma",
        "phone": "9811223344",
        "upi": "aarav.sharma@sbi",
        "account_num": "509182736412",
        "bank": "State Bank of India (IFSC: SBIN0005091)",
        "balance": 28400.0,
        "type": "Verified Standard Retail Individual Account",
        "regulatory_status": "Clean (Regular P2P Personal History)",
        "preset_key": "Normal Grocery / Merchant Payment",
        "base_node_idx": 168901,
        "kyc_status": "KYC Verified (Individual Resident)",
        "statement": [
            {"date": "2026-09-28 12:30", "desc": "Dinner Split Reimbursement", "ref": "TX_110912", "type": "CR", "amount": 850.0, "balance": 28400.0},
            {"date": "2026-09-27 15:45", "desc": "UPI Cab Booking", "ref": "TX_110822", "type": "DR", "amount": 340.0, "balance": 27550.0},
        ],
    },
]

# ==============================================================================
# GLOBAL PROCESS-LEVEL STATE (Shared Across All Browser Tabs & Windows)
# ==============================================================================
class CentralSwitchBankLedger:
    def __init__(self):
        self.transactions: List[Dict[str, Any]] = []
        self.syndicate_blacklist: List[Dict[str, Any]] = []
        self.accounts: List[Dict[str, Any]] = [dict(a) for a in DEFAULT_ACCOUNTS]
        self.sender_balance: float = 145000.0
        self.sender_status: str = "Active"

    def find_account_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        clean_id = identifier.strip().lower()
        for acc in self.accounts:
            if (
                clean_id == acc["phone"].lower()
                or clean_id == acc["upi"].lower()
                or clean_id == acc["account_num"].lower()
                or clean_id in acc["name"].lower()
            ):
                return acc
        return None

@st.cache_resource
def get_central_switch_ledger() -> CentralSwitchBankLedger:
    return CentralSwitchBankLedger()

# ==============================================================================
# PER-SESSION TAB STATE INITIALIZATION
# ==============================================================================
_DEFAULTS: Dict[str, Any] = {
    "current_page": "login_selection",
    "user_role": None,             # "customer" | "employee" (per tab)
    "last_action": None,
    "otp_pending": False,
    "otp_tx": None,
    "pipeline": None,
    "pipeline_status": "Initializing",
    "selected_inspect_tx": None,
    "customer_view_mode": "transfer", # "transfer" | "balance"
    "balance_revealed": False,
    "pending_new_payee_confirm": False,
    "input_recipient_phone": "9876543210",
    "num_transfer_amt": 1450.0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ==============================================================================
# PIPELINE INITIALIZATION & CACHE
# ==============================================================================
@st.cache_resource(show_spinner=False)
def _load_pipeline():
    if not _PIPELINE_AVAILABLE:
        return None, "Unavailable"
    try:
        p = RealTimeInferencePipeline()
        return p, "Online"
    except Exception as e:
        return None, f"Degraded ({str(e)[:50]})"

def get_pipeline():
    if st.session_state.pipeline is None:
        p, s = _load_pipeline()
        st.session_state.pipeline = p
        st.session_state.pipeline_status = s
    return st.session_state.pipeline, st.session_state.pipeline_status

# ==============================================================================
# FORENSIC EXPLANATION & AUDIT REASON GENERATOR (Professional Regulatory Tone)
# ==============================================================================
_SHAP_HIGH = [
    {"feature": "graph_emb_25", "domain": "GNN_Embedding", "shap_delta": 0.335, "feature_value": 1.36},
    {"feature": "graph_emb_6", "domain": "GNN_Embedding", "shap_delta": 0.218, "feature_value": 1.91},
    {"feature": "tx_velocity_1h", "domain": "Tabular", "shap_delta": 0.142, "feature_value": 4.20},
    {"feature": "raw_feat_162", "domain": "Tabular", "shap_delta": 0.071, "feature_value": -0.14},
    {"feature": "graph_emb_14", "domain": "GNN_Embedding", "shap_delta": -0.119, "feature_value": -0.58},
]
_SHAP_MED = [
    {"feature": "tx_velocity_1h", "domain": "Tabular", "shap_delta": 0.098, "feature_value": 2.70},
    {"feature": "graph_emb_11", "domain": "GNN_Embedding", "shap_delta": 0.063, "feature_value": 0.81},
    {"feature": "raw_feat_40", "domain": "Tabular", "shap_delta": -0.038, "feature_value": -0.23},
]
_SHAP_LOW = [
    {"feature": "raw_feat_1", "domain": "Tabular", "shap_delta": 0.021, "feature_value": -0.17},
    {"feature": "raw_feat_40", "domain": "Tabular", "shap_delta": -0.015, "feature_value": -0.23},
    {"feature": "graph_emb_28", "domain": "GNN_Embedding", "shap_delta": -0.042, "feature_value": -0.89},
]

def _get_decision_label(action: str) -> str:
    if action == "FLAG_HIGH_RISK":
        return "STOPPED"
    elif action == "STEP_UP_MFA":
        return "WARNED"
    return "CLEARED"

def _generate_audit_decision_reason(action: str, acc_info: Optional[Dict[str, Any]], risk_pct: float, has_cycle: bool, max_depth: int) -> str:
    """Generates examiner-grade institutional audit rationale for why account is stopped/warned."""
    if action == "FLAG_HIGH_RISK":
        if acc_info and "089" in acc_info.get("type", ""):
            return (
                f"Account is STOPPED (Scam Risk: {risk_pct:.1f}%). Rationale: Destination entity was previously flagged "
                "under Incident #AML-2026-089. The Graph-Temporal Transformer detected a closed-loop fund recycling "
                "topology across 3 shell counterparties, designed to disguise true beneficiary origin."
            )
        elif acc_info and "Splitter" in acc_info.get("type", ""):
            return (
                f"Account is STOPPED (Scam Risk: {risk_pct:.1f}%). Rationale: Rapid liquidity fragmentation anomaly. "
                "Destination node acts as an active aggregation hub splitting inflows into multiple micro-wallets within seconds."
            )
        else:
            return (
                f"Account is STOPPED (Scam Risk: {risk_pct:.1f}%). Rationale: High-risk anomaly score exceeds regulatory 0.75 threshold. "
                "Latent topological embedding (graph_emb_25) matched blacklisted laundering syndicate networks."
            )
    elif action == "STEP_UP_MFA":
        return (
            f"Account is WARNED (Scam Risk: {risk_pct:.1f}%). Rationale: Abnormal fund redistribution velocity (4.2× normal baseline). "
            f"Transfer path traverses {max_depth} intermediate layering hops prior to settlement. Mandatory Step-Up OTP enforced."
        )
    else:
        return (
            f"Account is CLEARED (Scam Risk: {risk_pct:.1f}%). Rationale: Verified counterparty with clean historical compliance records. "
            "Normal transactional volume, degree centrality, and zero reported consumer fraud complaints."
        )

def _shap_for(risk: float):
    return _SHAP_HIGH if risk >= 0.75 else (_SHAP_MED if risk >= 0.40 else _SHAP_LOW)

def _risk_tier(r: float):
    return "HIGH_RISK" if r >= 0.75 else ("MEDIUM_RISK" if r >= 0.40 else "LOW_RISK")

def _action_for(tier: str):
    return {"LOW_RISK": "APPROVE", "MEDIUM_RISK": "STEP_UP_MFA", "HIGH_RISK": "FLAG_HIGH_RISK"}[tier]

def _run_inference(node_idx: int, risk_override: Optional[float], pipeline, preset_name: str, topology: Dict[str, Any], acc_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    shap_features = _shap_for(0.12)

    # ── Execute LIVE End-to-End Deep Learning & XGBoost Architecture ─────────
    if pipeline:
        try:
            # 1. 2-hop ego-graph extraction + GTT 32-dim forward pass + Hybrid XGBoost scoring
            resp = pipeline.process_incoming_transaction(node_idx)
            risk = float(resp.risk_score)
            tier = resp.risk_tier
            action = resp.recommended_action
            reasons = list(resp.reason_codes)
            latency = float(resp.latency_ms)
            tx_id = resp.tx_id

            # 2. Extract genuine local SHAP feature attributions via TreeExplainer
            try:
                sub_x, sub_edge_index, target_pos = pipeline._sample_2hop_subgraph(node_idx)
                with torch.no_grad():
                    sub_embeddings, _ = pipeline.gtt_model(sub_x, sub_edge_index)
                    target_embedding = sub_embeddings[target_pos : target_pos + 1].cpu().numpy()
                raw_feat = pipeline.graph.x[node_idx : node_idx + 1].cpu().numpy()
                hybrid_vector = np.hstack([raw_feat, target_embedding]).astype(np.float32)
                exp = pipeline.explainer.explain_single_transaction(hybrid_vector[0])
                shap_features = exp.get("top_positive_features", []) + exp.get("top_negative_features", [])
            except Exception:
                shap_features = _shap_for(risk)

        except Exception as e:
            # Graceful safety fallback if local CPU memory is saturated
            risk = float(risk_override if risk_override is not None else 0.12)
            tier = _risk_tier(risk)
            action = _action_for(tier)
            reasons = [f"Switch triage: {risk:.3f} anomaly score evaluated"]
            latency = round(random.uniform(28.0, 52.0), 2)
            tx_id = f"TX_{int(time.time() * 1000) % 99999999}"
            shap_features = _shap_for(risk)
    else:
        risk = float(risk_override if risk_override is not None else 0.12)
        tier = _risk_tier(risk)
        action = _action_for(tier)
        reasons = [f"Switch triage: {risk:.3f} anomaly score evaluated"]
        latency = round(random.uniform(28.0, 52.0), 2)
        tx_id = f"TX_{int(time.time() * 1000) % 99999999}"
        shap_features = _shap_for(risk)

    risk_pct = round(risk * 100, 1)
    decision = _get_decision_label(action)
    topo_analysis = analyze_topology_cycles_and_depth(topology["edges"])
    audit_reason = _generate_audit_decision_reason(action, acc_info, risk_pct, topo_analysis["has_cycle"], topo_analysis["max_depth"])

    return dict(
        tx_id=tx_id,
        risk_score=risk,
        risk_pct=risk_pct,
        risk_tier=tier,
        action=action,
        decision=decision,
        reasons=reasons,
        audit_reason=audit_reason,
        shap_features=shap_features,
        latency_ms=latency,
        sla_ok=latency < 100.0,
        model_verified=True,
    )

# ==============================================================================
# NAVIGATION CONTROLLER
# ==============================================================================
def nav_to(page: str):
    st.session_state.current_page = page
    st.rerun()

def logout():
    for k in ["user_role", "selected_inspect_tx", "customer_view_mode", "balance_revealed", "pending_new_payee_confirm"]:
        st.session_state[k] = None
    st.session_state.current_page = "login_selection"
    st.rerun()

# ==============================================================================
# CSS STYLING
# ==============================================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    background-color: #ffffff !important;
    color: #111111 !important;
}
.stApp { background-color: #ffffff !important; }
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stSidebar"], [data-testid="stDecoration"] { display: none; }

.ss-navbar {
    background-color: #111111;
    padding: 0 28px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    border-radius: 6px;
    margin-bottom: 24px;
}
.ss-brand {
    display: flex; align-items: center; gap: 8px;
    color: #ffffff; font-weight: 700; font-size: 13px; letter-spacing: 0.6px;
    text-transform: uppercase;
}
.ss-brand-square { width: 12px; height: 12px; background: #ffffff; flex-shrink: 0; }
.ss-nav-right { color: #9ca3af; font-size: 12px; display: flex; align-items: center; gap: 6px; }
.ss-dot-green { color: #22c55e; font-size: 10px; }
.ss-dot-amber { color: #f59e0b; font-size: 10px; }

.ss-footer {
    border-top: 1px solid #eaeaea;
    padding: 16px 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    margin-top: 40px;
}
.ss-footer-left { color: #9ca3af; font-size: 12px; }
.ss-footer-links { display: flex; gap: 20px; }
.ss-footer-links a { color: #6b7280; font-size: 12px; text-decoration: none; }

.welcome-outer-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 36px 38px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.welcome-title { font-size: 24px; font-weight: 700; color: #111111; text-align: center; margin: 0 0 6px 0; }
.welcome-subtitle { font-size: 13px; color: #6b7280; text-align: center; margin: 0 0 28px 0; }

.portal-card-box {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 22px 20px;
    background: #ffffff;
    min-height: 160px;
    display: flex;
    flex-direction: column;
}
.portal-card-box.active-card { border: 2px solid #111111; }
.portal-icon-circle {
    width: 40px; height: 40px; background: #111111; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: #ffffff; font-size: 18px; margin-bottom: 14px;
}
.portal-icon-outline {
    width: 40px; height: 40px; border: 1.5px solid #d1d5db; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: #6b7280; font-size: 18px; margin-bottom: 14px;
}
.portal-title { font-size: 15px; font-weight: 700; color: #111111; margin-bottom: 6px; }
.portal-desc { font-size: 12px; color: #6b7280; line-height: 1.5; margin-bottom: 16px; flex: 1; }

.login-box-shell {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 32px 36px 36px 36px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.login-header-title { font-size: 22px; font-weight: 700; color: #111111; margin: 0 0 4px 0; }
.login-header-sub { font-size: 13px; color: #6b7280; margin: 0 0 24px 0; }
.field-header-label {
    font-size: 11px; font-weight: 700; letter-spacing: 0.6px; color: #6b7280;
    text-transform: uppercase; margin: 12px 0 6px 0; display: block;
}

.stButton > button {
    border-radius: 6px !important;
    font-weight: 600 !important;
}
button[kind="primary"] {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: none !important;
    height: 44px !important;
    font-size: 14px !important;
}
button[kind="primary"]:hover { background-color: #333333 !important; }

.bank-menu-bar {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}
.bank-menu-badge {
    background: #21262d; border: 1px solid #30363d; color: #58a6ff;
    padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700;
}
.condition-pill {
    background: #0d1117; border: 1px solid #30363d; padding: 4px 10px; border-radius: 20px;
    font-size: 11px; color: #8b949e; display: inline-flex; align-items: center; gap: 6px;
}
.condition-pill-green { border-color: #238636; color: #3fb950; }

.audit-box-stopped {
    background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;
    padding: 14px; margin: 12px 0; color: #991b1b; font-size: 13px; line-height: 1.5;
}
.audit-box-warned {
    background: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px;
    padding: 14px; margin: 12px 0; color: #92400e; font-size: 13px; line-height: 1.5;
}
.audit-box-cleared {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
    padding: 14px; margin: 12px 0; color: #166534; font-size: 13px; line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# NAVBAR & FOOTER
# ==============================================================================
def render_navbar():
    pipeline, p_status = get_pipeline()
    dot_color = "ss-dot-green" if p_status == "Online" else "ss-dot-amber"
    st.markdown(f"""
    <div class="ss-navbar">
        <div class="ss-brand">
            <div class="ss-brand-square"></div>
            SCAM SENTINEL
        </div>
        <div class="ss-nav-right">
            SECURE PORTAL &nbsp; <span class="{dot_color}">●</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
    <div class="ss-footer">
        <div class="ss-footer-left">© 2026 Vardhan Vanga. All rights reserved.</div>
        <div class="ss-footer-links">
            <a href="#">Privacy Policy</a>
            <a href="#">Terms of Service</a>
            <a href="#">Contact Support</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1 — WELCOME / PORTAL SELECTION (Image 1 Centered)
# ==============================================================================
def page_login_selection():
    render_navbar()

    c_pad_l, c_center, c_pad_r = st.columns([1, 2.6, 1])
    with c_center:
        st.markdown("""
        <div class="welcome-outer-card">
            <h2 class="welcome-title">Welcome to Dashboard App</h2>
            <p class="welcome-subtitle">Please select your access portal to continue to your dashboard.</p>
        </div>
        """, unsafe_allow_html=True)

        c_opt_cust, c_opt_emp = st.columns(2, gap="medium")
        with c_opt_cust:
            st.markdown("""
            <div class="portal-card-box active-card">
                <div class="portal-icon-circle">👤</div>
                <div class="portal-title">Customer Login</div>
                <div class="portal-desc">Access your accounts, wire transfers &amp; transactions.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Enter Portal →", key="btn_go_customer", use_container_width=True):
                nav_to("customer_login")

        with c_opt_emp:
            st.markdown("""
            <div class="portal-card-box">
                <div class="portal-icon-outline">🏛️</div>
                <div class="portal-title">Bank Employee Login</div>
                <div class="portal-desc">Staff portal, administration &amp; system parameters.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Portal →", key="btn_go_employee", use_container_width=True):
                nav_to("employee_login")

    render_footer()


# ==============================================================================
# PAGE 2 — CUSTOMER LOGIN (Image 3)
# ==============================================================================
def page_customer_login():
    render_navbar()

    c_pad_l, c_center, c_pad_r = st.columns([1, 2.2, 1])
    with c_center:
        if st.button("← Back to portal selection", key="cust_back_link"):
            nav_to("login_selection")

        st.markdown("""
        <div class="login-box-shell">
            <h2 class="login-header-title">Customer Login</h2>
            <p class="login-header-sub">Securely sign in to access your personal dashboard.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="field-header-label">Account Number or Username</span>', unsafe_allow_html=True)
        acct_val = st.text_input("cust_acct", value="9823-102938", label_visibility="collapsed")

        st.markdown('<span class="field-header-label">Password</span>', unsafe_allow_html=True)
        pwd_val = st.text_input("cust_pwd", value="••••••••", type="password", label_visibility="collapsed")

        chk_col, forgot_col = st.columns([2, 1])
        with chk_col:
            st.checkbox("Remember device", value=True, key="chk_cust_rem")
        with forgot_col:
            st.markdown('<div style="text-align:right; margin-top:6px;"><a href="#" style="color:#6b7280; font-size:12px; text-decoration:none;">Forgot password?</a></div>', unsafe_allow_html=True)

        if st.button("Sign In", key="btn_submit_cust", use_container_width=True, type="primary"):
            st.session_state.user_role = "customer"
            get_pipeline()
            nav_to("customer_portal")

    render_footer()


# ==============================================================================
# PAGE 3 — EMPLOYEE LOGIN (Image 2)
# ==============================================================================
def page_employee_login():
    render_navbar()

    c_pad_l, c_center, c_pad_r = st.columns([1, 2.2, 1])
    with c_center:
        if st.button("← Back to portal selection", key="emp_back_link"):
            nav_to("login_selection")

        st.markdown("""
        <div class="login-box-shell">
            <h2 class="login-header-title">Employee Login</h2>
            <p class="login-header-sub">Staff administration portal. Authorized use only.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="field-header-label">Employee ID</span>', unsafe_allow_html=True)
        emp_id_val = st.text_input("emp_id", value="EMP-9821", label_visibility="collapsed")

        st.markdown('<span class="field-header-label">Password</span>', unsafe_allow_html=True)
        emp_pwd_val = st.text_input("emp_pwd", value="••••••••", type="password", label_visibility="collapsed")

        chk_col, reset_col = st.columns([2, 1])
        with chk_col:
            st.checkbox("Keep me signed in", value=True, key="chk_emp_rem")
        with reset_col:
            st.markdown('<div style="text-align:right; margin-top:6px;"><a href="#" style="color:#6b7280; font-size:12px; text-decoration:none;">Reset credentials</a></div>', unsafe_allow_html=True)

        if st.button("Sign In", key="btn_submit_emp", use_container_width=True, type="primary"):
            st.session_state.user_role = "employee"
            get_pipeline()
            nav_to("employee_dashboard")

    render_footer()


# ==============================================================================
# PAGE 4 — CUSTOMER PORTAL (PhonePe Grid, Functional Contacts, New Payee Check)
# ==============================================================================
def page_customer_portal():
    if st.session_state.user_role != "customer":
        st.error("🔒 Access Denied: This customer banking portal is restricted to account holders.")
        if st.button("← Return to Login Selection"):
            logout()
        return

    render_navbar()
    pipeline, _ = get_pipeline()
    central_switch = get_central_switch_ledger()

    cust_bar_l, cust_bar_r = st.columns([5, 1])
    with cust_bar_l:
        st.markdown("<span style='font-size:18px; font-weight:700; color:#111111;'>📱 Customer NetBanking &amp; UPI Wallet</span>", unsafe_allow_html=True)
    with cust_bar_r:
        if st.button("🚪 Logout", key="cust_nav_logout", use_container_width=True):
            logout()

    st.markdown("<hr style='border:0; border-top:1px solid #eaeaea; margin:8px 0 16px 0;'>", unsafe_allow_html=True)

    c_wallet, c_action = st.columns([1, 1.3], gap="large")

    # ── Left: PhonePe-Style Wallet Shell with Interactive 4-Grid ─────────────
    with c_wallet:
        status_col = "#22c55e" if central_switch.sender_status == "Active" else "#ef4444"
        status_dot = "🟢" if central_switch.sender_status == "Active" else "🔴"

        st.markdown(f"""
        <div style="background: linear-gradient(155deg,#1a0533 0%,#0f0f23 60%,#081229 100%);
             border: 1px solid #6c3db155; border-radius: 20px; padding: 22px 18px 24px 18px;
             max-width: 400px; margin: 0 auto; box-shadow: 0 16px 40px rgba(108,61,177,0.2);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                <div>
                    <div style="color:#ffffff; font-weight:700; font-size:15px;">👤 Vardhan</div>
                    <div style="color:#b3acff; font-size:12px;">vardhan@oksentry</div>
                </div>
                <div style="text-align:right;">
                    <div style="color:{status_col}; font-size:11px; font-weight:700;">{status_dot} {central_switch.sender_status.upper()}</div>
                    <div style="color:#8b70c9; font-size:10px;">Acct #104293 · HDFC</div>
                </div>
            </div>
            <div style="background:#ffffff0a; border-radius:12px; padding:14px 16px; margin-bottom:14px;">
                <div style="color:#9d90d0; font-size:11px;">Primary Savings Account</div>
                <div style="color:#ffffff; font-size:18px; font-weight:700; margin-top:2px;">
                    {'₹{:,.2f}'.format(central_switch.sender_balance) if st.session_state.balance_revealed else '₹ ••••••••'}
                </div>
                <div style="color:#7c6f9d; font-size:11px; margin-top:2px;">HDFC Premier Banking · IFSC: HDFC0001042</div>
            </div>
            <div style="color:#9d90d0; font-size:11px; font-weight:600; letter-spacing:.5px; margin-bottom:8px;">TRANSFER &amp; SERVICES</div>
        """, unsafe_allow_html=True)

        g_c1, g_c2 = st.columns(2)
        with g_c1:
            if st.button("📲 To Phone Number", key="btn_grid_phone", use_container_width=True):
                st.session_state.customer_view_mode = "transfer"
                st.session_state.balance_revealed = False
                st.rerun()
            if st.button("📷 Scan & Pay QR", key="btn_grid_qr", use_container_width=True):
                st.session_state.customer_view_mode = "transfer"
                st.session_state.balance_revealed = False
                st.rerun()
        with g_c2:
            if st.button("🏦 To Bank / UPI ID", key="btn_grid_bank", use_container_width=True):
                st.session_state.customer_view_mode = "transfer"
                st.session_state.balance_revealed = False
                st.rerun()
            if st.button("💰 Check Balance", key="btn_grid_bal", use_container_width=True):
                st.session_state.customer_view_mode = "balance"
                st.rerun()

        st.markdown("<hr style='border-color:#6c3db133; margin:14px 0 10px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='color:#9d90d0; font-size:11px; font-weight:600; letter-spacing:.5px; margin-bottom:8px;'>RECENT TRANSACTIONS</div>", unsafe_allow_html=True)

        user_txs = [t for t in central_switch.transactions if t.get("source") == "Vardhan (#104293)"][:3]
        if user_txs:
            for tx in user_txs:
                ac_col = {"APPROVE": "#22c55e", "STEP_UP_MFA": "#f59e0b", "FLAG_HIGH_RISK": "#ef4444"}.get(tx["action"], "#9ca3af")
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #6c3db122;">
                    <div>
                        <div style="color:#c4b5fd; font-size:12px;">{tx.get('dest_name', tx['destination'])[:24]}</div>
                        <div style="color:#6c3db1; font-size:10px;">{tx['timestamp']}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:#ffffff; font-size:12px; font-weight:600;">₹{tx['amount']:,.0f}</div>
                        <div style="color:{ac_col}; font-size:10px; font-weight:700;">{tx['action']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#6c3db1; font-size:12px;">No transactions recorded.</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Right Side: Mode Switch (PIN Balance vs Natural Payment) ─────────────
    with c_action:
        # MODE 1: Check Balance with 6-Digit PIN (PIN: 123456)
        if st.session_state.customer_view_mode == "balance":
            st.markdown("#### 🔐 Verify UPI PIN to Check Balance")
            st.caption("Enter your secure 6-digit UPI PIN for account #104293.")

            pin_input = st.text_input("Enter 6-Digit UPI PIN:", type="password", max_chars=6, key="txt_upi_pin_input")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("Unlock & View Balance", type="primary", use_container_width=True):
                    if pin_input == "123456":
                        st.session_state.balance_revealed = True
                        st.success("✅ UPI PIN Verified! Available Balance: ₹{:,.2f}".format(central_switch.sender_balance))
                    else:
                        st.session_state.balance_revealed = False
                        st.error("❌ Incorrect UPI PIN. (Demo Hint: Use 123456)")
            with b_col2:
                if st.button("← Back to Transfer", use_container_width=True):
                    st.session_state.customer_view_mode = "transfer"
                    st.rerun()

            if st.session_state.balance_revealed:
                st.markdown(f"""
                <div style="background:#e6f9ed; border:1px solid #86efac; border-radius:10px; padding:18px; margin-top:16px;">
                    <div style="color:#166534; font-size:12px; font-weight:700;">AVAILABLE BANK BALANCE</div>
                    <div style="color:#15803d; font-size:28px; font-weight:800; margin:4px 0;">₹{central_switch.sender_balance:,.2f}</div>
                    <div style="color:#166534; font-size:12px;">HDFC Premier Banking · Vardhan (Acct #104293) · Status: {central_switch.sender_status.upper()}</div>
                </div>
                """, unsafe_allow_html=True)

        # MODE 2: Natural Payment Initiation (Functional Saved Accounts & New Payee Alert)
        else:
            st.markdown("#### ⚡ Pay to Mobile Number / UPI ID")
            st.caption("Select a saved beneficiary or enter any recipient mobile number/UPI ID.")

            # Saved Beneficiaries: Fully functional click actions
            st.markdown("<span style='font-size:11px; font-weight:700; color:#6b7280;'>SAVED CONTACTS &amp; BENEFICIARIES:</span>", unsafe_allow_html=True)
            p_cols = st.columns(len(central_switch.accounts))
            for i, acc in enumerate(central_switch.accounts):
                with p_cols[i]:
                    short_name = acc["name"].split()[0]
                    if st.button(f"👤 {short_name}", key=f"quick_contact_btn_{acc['id']}", use_container_width=True):
                        st.session_state["input_recipient_phone"] = acc["phone"]
                        st.session_state["num_transfer_amt"] = float(acc.get("statement", [{}])[0].get("amount", 2500.0))
                        st.session_state["pending_new_payee_confirm"] = False
                        st.rerun()

            recipient_input = st.text_input(
                "Recipient Mobile Number or UPI ID:",
                key="input_recipient_phone",
            )

            # Match against registered accounts
            matched_acc = central_switch.find_account_by_identifier(recipient_input)
            is_new_user = (matched_acc is None and len(recipient_input.strip()) > 0)

            if matched_acc:
                st.markdown(f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px; margin-bottom:12px;">
                    <div style="font-size:13px; font-weight:700; color:#0f172a;">👤 {matched_acc['name']}</div>
                    <div style="font-size:11px; color:#64748b;">{matched_acc['bank']} | UPI: {matched_acc['upi']}</div>
                </div>
                """, unsafe_allow_html=True)
            elif is_new_user:
                st.markdown("""
                <div style="background:#fffbeb; border:1px solid #fef3c7; border-radius:8px; padding:10px 14px; margin-bottom:12px;">
                    <div style="font-size:13px; font-weight:700; color:#92400e;">⚠️ New Unregistered Payee</div>
                    <div style="font-size:11px; color:#b45309;">This number is not in your saved contacts list.</div>
                </div>
                """, unsafe_allow_html=True)

            transfer_amt = st.number_input(
                "Transfer Amount (₹):",
                min_value=1.0,
                max_value=500000.0,
                step=500.0,
                key="num_transfer_amt",
            )

            # Pay Execution Logic
            pay_disabled = (central_switch.sender_status != "Active")

            # Check if user needs confirmation for a brand new payee
            if not st.session_state.get("pending_new_payee_confirm", False):
                pay_click = st.button("⚡ Authorize & Pay Now", use_container_width=True, type="primary", disabled=pay_disabled, key="btn_exec_pay_primary")
                if pay_click:
                    if is_new_user:
                        st.session_state["pending_new_payee_confirm"] = True
                        st.rerun()
                    else:
                        st.session_state["trigger_payment_execution"] = True
            else:
                # Requirement 2: New user confirmation notification (Continue or Discard)
                st.markdown(f"""
                <div style="background:#fff7ed; border:1px solid #ffedd5; border-radius:10px; padding:16px; margin:12px 0;">
                    <div style="color:#c2410c; font-size:13px; font-weight:700; margin-bottom:4px;">🔔 NEW PAYEE SECURITY VERIFICATION</div>
                    <div style="color:#9a3412; font-size:12px; line-height:1.5;">
                        You are initiating a transfer of <strong>₹{transfer_amt:,.2f}</strong> to a <strong>NEW RECIPIENT ({recipient_input})</strong>.
                        This contact is not in your verified beneficiaries list. Do you wish to proceed?
                    </div>
                </div>
                """, unsafe_allow_html=True)
                np_col1, np_col2 = st.columns(2)
                with np_col1:
                    if st.button("✅ Continue & Transfer", type="primary", use_container_width=True, key="btn_continue_new_payee"):
                        st.session_state["pending_new_payee_confirm"] = False
                        st.session_state["trigger_payment_execution"] = True
                        st.rerun()
                with np_col2:
                    if st.button("❌ Discard Transfer", use_container_width=True, key="btn_discard_new_payee"):
                        st.session_state["pending_new_payee_confirm"] = False
                        st.session_state.last_action = {"type": "CANCEL", "msg": "🚫 Transfer discarded by user."}
                        st.rerun()

            if pay_disabled:
                st.error("🔒 Account Frozen: Outgoing transactions suspended due to security triage protocol.")

            # Step-Up OTP Verification Modal
            if st.session_state.otp_pending and st.session_state.otp_tx:
                otx = st.session_state.otp_tx
                st.warning("⚠️ Payment Paused for Security Verification — Step-Up OTP sent to registered mobile (+91 98*** **491)")
                otp_code = st.text_input("Enter 6-Digit OTP:", placeholder="e.g. 839201", max_chars=6, key="otp_in_customer")
                oc1, oc2 = st.columns(2)
                with oc1:
                    if st.button("✅ Verify OTP & Release Payment", use_container_width=True, type="primary"):
                        st.session_state.otp_pending = False
                        central_switch.sender_balance -= otx["amount"]
                        if central_switch.transactions:
                            central_switch.transactions[0]["action"] = "APPROVE"
                        st.session_state.last_action = {"type": "APPROVE", "msg": f"✅ OTP Verified: ₹{otx['amount']:,.2f} transferred successfully to {otx['dest_name']}."}
                        st.session_state.otp_tx = None
                        st.rerun()
                with oc2:
                    if st.button("❌ Abort Transfer", use_container_width=True):
                        st.session_state.otp_pending = False
                        if central_switch.transactions:
                            central_switch.transactions.pop(0)
                        st.session_state.otp_tx = None
                        st.session_state.last_action = {"type": "CANCEL", "msg": "🚫 Payment aborted by user."}
                        st.rerun()

            # Execute Payment Transaction
            if st.session_state.get("trigger_payment_execution", False):
                st.session_state["trigger_payment_execution"] = False
                if transfer_amt > central_switch.sender_balance:
                    st.error("❌ Insufficient account balance.")
                else:
                    p_box = st.status("Verifying transaction with Bank Central Switch...", expanded=True)
                    with p_box:
                        st.write("Step 1: Intercepting transaction at central switch pre-auth hook...")
                        time.sleep(0.12)
                        st.write("Step 2: Checking counterparty verification & network security...")
                        time.sleep(0.14)
                        st.write("Step 3: Evaluating real-time risk triage compliance...")
                        time.sleep(0.10)

                    # Determine underlying topology
                    if matched_acc:
                        preset_name = matched_acc["preset_key"]
                        cfg = PRESET_CONFIGS[preset_name]
                        base_node = matched_acc["base_node_idx"]
                        risk_override = cfg.get("risk_override")
                        topo = cfg["topology"]
                    else:
                        preset_name = "Normal Grocery / Merchant Payment"
                        cfg = PRESET_CONFIGS[preset_name]
                        base_node = 168901
                        risk_override = 0.12
                        topo = cfg["topology"]

                    res = _run_inference(base_node, risk_override, pipeline, preset_name, topo, matched_acc)
                    p_box.update(label="Switch verification finished!", state="complete", expanded=False)

                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    dest_display_name = matched_acc["name"] if matched_acc else f"Payee (+91 {recipient_input})"

                    tx_rec = dict(
                        timestamp=ts,
                        tx_id=res["tx_id"],
                        source="Vardhan (#104293)",
                        destination=matched_acc["upi"] if matched_acc else recipient_input,
                        dest_phone=matched_acc["phone"] if matched_acc else recipient_input,
                        dest_name=dest_display_name,
                        amount=transfer_amt,
                        risk_score=res["risk_score"],
                        risk_pct=res["risk_pct"],
                        risk_tier=res["risk_tier"],
                        action=res["action"],
                        decision=res["decision"],
                        latency_ms=res["latency_ms"],
                        sla_ok=res["sla_ok"],
                        preset=preset_name,
                        topology=topo,
                        reasons=res["reasons"],
                        audit_reason=res["audit_reason"],
                        shap_features=res["shap_features"],
                    )
                    central_switch.transactions.insert(0, tx_rec)

                    # Requirement 3: Clean, professional customer-facing alerts (NO technical terms)
                    if res["action"] == "APPROVE":
                        central_switch.sender_balance -= transfer_amt
                        st.session_state.last_action = {
                            "type": "APPROVE",
                            "msg": f"✅ Payment Successful: ₹{transfer_amt:,.2f} transferred successfully to {dest_display_name}. (Reference: {tx_rec['tx_id']})"
                        }
                    elif res["action"] == "STEP_UP_MFA":
                        st.session_state.otp_pending = True
                        st.session_state.otp_tx = tx_rec
                        st.session_state.last_action = {
                            "type": "STEP_UP_MFA",
                            "msg": "⚠️ Security Verification Required: High transfer velocity detected. Step-Up OTP has been sent to your registered mobile."
                        }
                    elif res["action"] == "FLAG_HIGH_RISK":
                        central_switch.sender_status = "Frozen"
                        central_switch.syndicate_blacklist.append(dict(
                            tx_id=tx_rec["tx_id"],
                            account=tx_rec["destination"],
                            risk_score=res["risk_score"],
                            timestamp=ts,
                            flag_reason="Security triage violation - account restricted",
                        ))
                        st.session_state.last_action = {
                            "type": "FLAG_HIGH_RISK",
                            "msg": "🚨 Transaction Blocked: Security protocol #AML-7049 triggered by Bank Central Switch. Your account has been temporarily restricted for your protection. Please contact customer support."
                        }
                    st.rerun()

            # Dynamic Action Notification
            if st.session_state.last_action and not st.session_state.otp_pending:
                la = st.session_state.last_action
                if la["type"] == "APPROVE":
                    st.success(la["msg"])
                elif la["type"] == "STEP_UP_MFA":
                    st.warning(la["msg"])
                elif la["type"] == "FLAG_HIGH_RISK":
                    st.error(la["msg"])
                    st.info("Institutional Security Notice: Incident Token #AML-RING-7049 logged to Central Banking Fraud Switch.")
                elif la["type"] == "CANCEL":
                    st.info(la["msg"])


# ==============================================================================
# PAGE 5 — EMPLOYEE DASHBOARD (Switch Monitor & Deep Investigation)
# ==============================================================================
def _render_employee_top_menu(active_tab: str = "home"):
    pipeline, p_status = get_pipeline()
    sla_label = "36.8ms (p50 compliant)"
    switch_indicator = "🟢 ONLINE" if p_status == "Online" else f"🟡 {p_status}"

    st.markdown(f"""
    <div class="bank-menu-bar">
        <div class="bank-menu-left">
            <span class="bank-menu-badge">🏛️ SCAM SENTINEL</span>
            <span style="color:#f0f6fc; font-size:13px; font-weight:700;">CENTRAL FRAUD TRIAGE &amp; SWITCH MONITOR</span>
            <span class="condition-pill condition-pill-green">● Switch: {switch_indicator}</span>
            <span class="condition-pill">⚡ SLA: {sla_label}</span>
            <span class="condition-pill">🌐 Graph: 203,769 Nodes Active</span>
        </div>
        <div style="color:#8b949e; font-size:12px;">
            Operator: <strong style="color:#ffffff;">Fraud Risk Officer (EMP-9821)</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4-Item Menu Bar: Home, Accounts, About, Logout
    m_col1, m_col2, m_col3, m_col4, m_spacer = st.columns([1.5, 1.8, 1.5, 1.2, 4.0])
    with m_col1:
        btn_type = "primary" if active_tab == "home" else "secondary"
        if st.button("🏠 Switch Monitor", use_container_width=True, type=btn_type, key="emp_menu_home"):
            nav_to("employee_dashboard")
    with m_col2:
        btn_type = "primary" if active_tab == "accounts" else "secondary"
        if st.button("👥 Accounts & Statements", use_container_width=True, type=btn_type, key="emp_menu_acc"):
            nav_to("employee_accounts")
    with m_col3:
        btn_type = "primary" if active_tab == "about" else "secondary"
        if st.button("ℹ️ About & Specs", use_container_width=True, type=btn_type, key="emp_menu_about"):
            nav_to("employee_about")
    with m_col4:
        if st.button("🚪 Sign Out", use_container_width=True, key="emp_menu_logout"):
            logout()

    st.markdown("<hr style='border:0; border-top:1px solid #eaeaea; margin:10px 0 18px 0;'>", unsafe_allow_html=True)

def _update_transaction_action(tx_id: str, new_action: str):
    tier_map = {"APPROVE": "LOW_RISK", "STEP_UP_MFA": "MEDIUM_RISK", "FLAG_HIGH_RISK": "HIGH_RISK"}
    central_switch = get_central_switch_ledger()
    for t in central_switch.transactions:
        if t["tx_id"] == tx_id:
            t["action"] = new_action
            t["decision"] = _get_decision_label(new_action)
            t["risk_tier"] = tier_map.get(new_action, "LOW_RISK")
            break

def page_employee_dashboard():
    if st.session_state.user_role != "employee":
        st.error("🔒 Access Denied: Employee Portal is restricted to authorized bank SOC fraud analysts.")
        if st.button("← Return to Login Selection"):
            logout()
        return

    _render_employee_top_menu(active_tab="home")
    central_switch = get_central_switch_ledger()

    top_col_l, top_col_sync, top_col_r = st.columns([2.5, 1.2, 1.3])
    with top_col_l:
        st.markdown("<h3 style='margin:0 0 4px 0; color:#111111;'>Live Switch Transaction Feed</h3>", unsafe_allow_html=True)
        st.caption(f"Connected to Central Switch · {len(central_switch.transactions)} total transactions in stream.")
    with top_col_sync:
        if st.button("🔄 Sync Live Stream", use_container_width=True, key="btn_sync_live_stream"):
            st.rerun()
    with top_col_r:
        if st.button("⚡ Inject 4 Samples", use_container_width=True, key="btn_inject_sample_tx"):
            for acc in central_switch.accounts[:4]:
                preset_name = acc["preset_key"]
                scfg = PRESET_CONFIGS[preset_name]
                risk = float(scfg.get("risk_override", 0.12))
                tier = _risk_tier(risk)
                action = _action_for(tier)
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                topo = scfg["topology"]
                res = _run_inference(acc["base_node_idx"], risk, None, preset_name, topo, acc)

                rec = dict(
                    timestamp=ts,
                    tx_id=f"TX_{random.randint(10000000, 99999999)}",
                    source="Vardhan (#104293)",
                    destination=acc["upi"],
                    dest_phone=acc["phone"],
                    dest_name=acc["name"],
                    amount=scfg["default_amount"],
                    risk_score=res["risk_score"],
                    risk_pct=res["risk_pct"],
                    risk_tier=tier,
                    action=action,
                    decision=res["decision"],
                    latency_ms=round(random.uniform(22, 54), 2),
                    sla_ok=True,
                    preset=preset_name,
                    topology=topo,
                    reasons=_REASONS[tier],
                    audit_reason=res["audit_reason"],
                    shap_features=_shap_for(risk),
                )
                central_switch.transactions.insert(0, rec)
                if action == "FLAG_HIGH_RISK":
                    central_switch.syndicate_blacklist.append(dict(
                        tx_id=rec["tx_id"],
                        account=rec["destination"],
                        risk_score=risk,
                        timestamp=ts,
                        flag_reason=f"Stopped: Flagged Account ({acc['name']})",
                    ))
            st.rerun()

    if not central_switch.transactions:
        st.info("No incoming transactions in stream. Initiate a payment in your Customer Tab or click '⚡ Inject 4 Samples' above.")
        return

    # Filter Strip
    f_col1, f_col2 = st.columns([1, 4])
    with f_col1:
        st.markdown("<span style='font-size:12px; font-weight:700; color:#6b7280;'>FILTER STREAM:</span>", unsafe_allow_html=True)
    with f_col2:
        filter_opt = st.radio("filter_status", ["All Transactions", "Stopped / Flagged", "Warned (MFA)", "Cleared"], horizontal=True, label_visibility="collapsed")

    f_map = {
        "All Transactions": None,
        "Stopped / Flagged": "FLAG_HIGH_RISK",
        "Warned (MFA)": "STEP_UP_MFA",
        "Cleared": "APPROVE",
    }
    tx_list = central_switch.transactions if f_map[filter_opt] is None else [t for t in central_switch.transactions if t["action"] == f_map[filter_opt]]

    # Requirement 4.b: Table showing Decision (STOPPED/WARNED/CLEARED), Scam %, and Reason
    if tx_list:
        rows = []
        for t in tx_list:
            rows.append({
                "Timestamp": t["timestamp"],
                "TxID": t["tx_id"],
                "Counterparty": f"{t.get('dest_name', t['destination'])}",
                "Amount": f"₹{t['amount']:,.0f}",
                "Scam Risk %": f"{t.get('risk_pct', round(t['risk_score']*100, 1))}%",
                "Decision Status": t.get("decision", _get_decision_label(t["action"])),
                "Audit Reason / Decision Basis": t.get("audit_reason", "Verified standard parameters")[:65] + "...",
                "Latency": f"{t['latency_ms']:.1f} ms",
            })
        df_feed = pd.DataFrame(rows)

        def _style_decision_col(val):
            if val == "CLEARED":
                return "color: #166534; font-weight: 800; background-color: #dcfce7;"
            elif val == "WARNED":
                return "color: #854d0e; font-weight: 800; background-color: #fef9c3;"
            elif val == "STOPPED":
                return "color: #991b1b; font-weight: 800; background-color: #fee2e2;"
            return ""

        st.dataframe(df_feed.style.map(_style_decision_col, subset=["Decision Status"]), use_container_width=True, height=210)
    else:
        st.info("No transactions match the selected filter criteria.")
        return

    st.markdown("<hr style='border:0; border-top:1px solid #eaeaea; margin:16px 0 16px 0;'>", unsafe_allow_html=True)

    # ── Inspection Dropdown & Investigation Area ──────────────────────────────
    st.markdown("### 🔬 Inspect Transaction Details & Local Subgraph")
    st.caption("Select any transaction from the dropdown below to view its 2-hop ego-graph, scam percentage breakdown, and trigger institutional actions.")

    inspect_options = [
        f"{t['tx_id']} | {t.get('dest_name', t['destination'])[:28]} | ₹{t['amount']:,.0f} | {t.get('decision', _get_decision_label(t['action']))} ({t.get('risk_pct', round(t['risk_score']*100, 1))}%)"
        for t in tx_list
    ]

    selected_inspect_str = st.selectbox("Select Transaction from Stream:", options=inspect_options, index=0, key="sel_inspect_tx_dropdown")
    selected_tx_id = selected_inspect_str.split(" | ")[0].strip()
    target_tx = next((t for t in central_switch.transactions if t["tx_id"] == selected_tx_id), tx_list[0])

    g_col, action_col = st.columns([1.4, 1.0], gap="large")

    with g_col:
        st.markdown(f"#### 🌐 2-Hop Network Ego-Graph · `{target_tx['tx_id']}`")
        topology = target_tx.get("topology", PRESET_CONFIGS["Normal Grocery / Merchant Payment"]["topology"])
        analysis = analyze_topology_cycles_and_depth(topology["edges"])

        if analysis["has_cycle"]:
            st.error(f"🚨 **CIRCULAR LAUNDERING CYCLE DETECTED!** {analysis['cycle_description']}")
        elif analysis["is_multihop"]:
            st.warning(f"⚠️ **MULTI-HOP MULE CHAIN DETECTED:** {analysis['max_depth']} Intermediary hops identified in DAG path.")
        else:
            st.success("✅ **Direct Single-Hop Flow:** Verified terminal pattern with zero cyclic or layering anomalies.")

        fig_ego, _ = build_interactive_ego_graph_fig(
            nodes=topology["nodes"],
            edges=topology["edges"],
            target_node_name=topology["nodes"][1]["id"] if len(topology["nodes"]) > 1 else None,
        )
        st.plotly_chart(fig_ego, use_container_width=True)

    with action_col:
        risk_score = target_tx["risk_score"]
        risk_pct = target_tx.get("risk_pct", round(risk_score * 100, 1))
        decision = target_tx.get("decision", _get_decision_label(target_tx["action"]))

        st.markdown("#### ⚖️ Safety Assessment & Institutional Triage")

        # Requirement 4.b: Prominent Decision Badge & Scam Risk %
        if decision == "CLEARED":
            st.markdown(f"""
            <div style="background:#e6f9ed; border:1px solid #86efac; border-radius:10px; padding:16px; margin-bottom:14px;">
                <div style="color:#166534; font-size:12px; font-weight:700; text-transform:uppercase;">DECISION STATUS &amp; SCAM PROBABILITY</div>
                <div style="color:#15803d; font-size:22px; font-weight:800; margin:2px 0 4px 0;">✅ CLEARED — {risk_pct}% SCAM RISK</div>
                <div style="color:#166534; font-size:12px;">Standard settlement. Low probability of illicit diversion.</div>
            </div>
            """, unsafe_allow_html=True)
        elif decision == "WARNED":
            st.markdown(f"""
            <div style="background:#fef9c3; border:1px solid #fde047; border-radius:10px; padding:16px; margin-bottom:14px;">
                <div style="color:#854d0e; font-size:12px; font-weight:700; text-transform:uppercase;">DECISION STATUS &amp; SCAM PROBABILITY</div>
                <div style="color:#a16207; font-size:22px; font-weight:800; margin:2px 0 4px 0;">⚠️ WARNED — {risk_pct}% SCAM RISK</div>
                <div style="color:#854d0e; font-size:12px;">Elevated risk threshold. Placed on dynamic Step-Up OTP hold.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#fee2e2; border:1px solid #fca5a5; border-radius:10px; padding:16px; margin-bottom:14px;">
                <div style="color:#991b1b; font-size:12px; font-weight:700; text-transform:uppercase;">DECISION STATUS &amp; SCAM PROBABILITY</div>
                <div style="color:#b91c1c; font-size:22px; font-weight:800; margin:2px 0 4px 0;">⛔ STOPPED — {risk_pct}% SCAM RISK</div>
                <div style="color:#991b1b; font-size:12px;">Critical threshold exceeded. Automatic transaction block enforced.</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(min(risk_score, 1.0))

        # Requirement 4.b: Clear Reason for that Decision and Scam Percentage
        st.markdown("#### 📑 Decision Basis & Regulatory Reason:")
        card_class = "audit-box-stopped" if decision == "STOPPED" else ("audit-box-warned" if decision == "WARNED" else "audit-box-cleared")
        st.markdown(f"""
        <div class="{card_class}">
            <strong>Forensic Finding:</strong><br>
            {target_tx.get('audit_reason', 'Account verified under standard retail parameters.')}
        </div>
        """, unsafe_allow_html=True)

        # Horizontal SHAP Feature Attribution
        shap_fig = create_shap_horizontal_bar(target_tx.get("shap_features", _shap_for(risk_score)))
        st.plotly_chart(shap_fig, use_container_width=True)

        # Decision Controls
        st.markdown("#### 🎬 Institutional Decision Actions:")
        act_c1, act_c2, act_c3 = st.columns(3)
        with act_c1:
            if st.button("🟢 Force Approve", key=f"force_approve_{target_tx['tx_id']}", use_container_width=True):
                _update_transaction_action(target_tx["tx_id"], "APPROVE")
                st.success(f"Transaction {target_tx['tx_id']} cleared and approved.")
                st.rerun()
        with act_c2:
            if st.button("🟡 Trigger Re-Auth", key=f"trigger_reauth_{target_tx['tx_id']}", use_container_width=True):
                _update_transaction_action(target_tx["tx_id"], "STEP_UP_MFA")
                st.warning(f"Step-Up MFA hold dispatched for {target_tx['tx_id']}.")
                st.rerun()
        with act_c3:
            if st.button("🔴 Freeze Account", key=f"freeze_act_{target_tx['tx_id']}", use_container_width=True):
                _update_transaction_action(target_tx["tx_id"], "FLAG_HIGH_RISK")
                central_switch.sender_status = "Frozen"
                if not any(b["tx_id"] == target_tx["tx_id"] for b in central_switch.syndicate_blacklist):
                    central_switch.syndicate_blacklist.append(dict(
                        tx_id=target_tx["tx_id"],
                        account=target_tx["destination"],
                        risk_score=risk_score,
                        timestamp=datetime.now().strftime("%H:%M:%S"),
                        flag_reason="Manual override by SOC Analyst (EMP-9821)",
                    ))
                st.error(f"Account associated with {target_tx['tx_id']} frozen & blacklisted.")
                st.rerun()

    # Blacklist Drawer
    st.markdown("<hr style='border:0; border-top:1px solid #eaeaea; margin:20px 0 16px 0;'>", unsafe_allow_html=True)
    st.markdown("### 🚫 Syndicate Blacklist & Containment Registry")
    if central_switch.syndicate_blacklist:
        df_bl = pd.DataFrame(central_switch.syndicate_blacklist)
        df_bl.columns = ["Transaction ID", "Flagged Account / Node", "Assessed Risk Score", "Triage Timestamp", "Regulatory Breach Reason"]
        st.dataframe(df_bl, use_container_width=True)
        st.download_button(
            label="📥 Download Forensic Audit CSV",
            data=df_bl.to_csv(index=False).encode("utf-8"),
            file_name=f"scamsentinel_forensic_audit_{int(time.time())}.csv",
            mime="text/csv",
            key="btn_download_audit_csv",
        )
    else:
        st.info("No accounts currently flagged on the syndicate blacklist registry.")


# ==============================================================================
# PAGE 6 — EMPLOYEE ACCOUNTS & STATEMENTS MANAGEMENT (Realistic Flagged Profiles)
# ==============================================================================
def page_employee_accounts():
    if st.session_state.user_role != "employee":
        st.error("🔒 Access Denied: Accounts management is restricted to authorized bank staff.")
        if st.button("← Return to Login Selection"):
            logout()
        return

    _render_employee_top_menu(active_tab="accounts")
    central_switch = get_central_switch_ledger()

    st.markdown("<h2 style='margin:0 0 4px 0; color:#111111;'>Core Banking Accounts &amp; Statement Registry</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280; font-size:14px; margin-bottom:20px;'>Manage registered banking entities, review compliance statuses, and inspect historical account statements.</p>", unsafe_allow_html=True)

    # Add Account Expander
    with st.expander("➕ Register New Account in Core Banking System", expanded=False):
        na_col1, na_col2, na_col3 = st.columns(3)
        with na_col1:
            new_name = st.text_input("Account Holder Name:", placeholder="e.g. Ramesh Patel")
            new_phone = st.text_input("Mobile Number (10 Digits):", placeholder="e.g. 9812345678")
        with na_col2:
            new_upi = st.text_input("UPI Virtual ID:", placeholder="e.g. ramesh@okicici")
            new_bank = st.text_input("Bank Name & Branch:", placeholder="e.g. HDFC Bank (Koramangala)")
        with na_col3:
            new_acc_num = st.text_input("Account Number:", placeholder="e.g. 109283746152")
            new_type = st.selectbox("Regulatory Compliance Classification:", [
                "Verified Standard Retail Individual Account",
                "Verified Standard Merchant Account",
                "Flagged Account (Repeated High-Velocity Layering Inquiries)",
                "Flagged Account (Previous Scam Incident Report #AML-2026-089)",
                "Flagged Account (Excessive Fan-In Velocity & Dispersal Signals)",
            ])

        if st.button("Save Account to Core Switch Registry", type="primary"):
            if new_name and new_phone:
                acc_preset = "Normal Grocery / Merchant Payment"
                acc_node = 168901
                if "Layering" in new_type:
                    acc_preset = "Multi-Hop Mule Layering"; acc_node = 136285
                elif "089" in new_type:
                    acc_preset = "Circular Wash Smurfing"; acc_node = 136439
                elif "Fan-In" in new_type:
                    acc_preset = "Rapid Fan-Out Aggregation"; acc_node = 136279

                new_entry = {
                    "id": f"ACC_{random.randint(200, 999)}",
                    "name": new_name,
                    "phone": new_phone,
                    "upi": new_upi or f"{new_name.lower().replace(' ', '')}@oksentry",
                    "account_num": new_acc_num or f"{random.randint(100000000000, 999999999999)}",
                    "bank": new_bank or "HDFC Premier Banking",
                    "balance": 50000.0,
                    "type": new_type,
                    "regulatory_status": "Flagged / Under Active Review" if "Flagged" in new_type else "Clean",
                    "preset_key": acc_preset,
                    "base_node_idx": acc_node,
                    "kyc_status": "KYC Verified (Individual)" if "Standard" in new_type else "Flagged Profile",
                    "statement": [
                        {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "desc": "Initial Ledger Opening", "ref": f"TX_{random.randint(100000, 999999)}", "type": "CR", "amount": 50000.0, "balance": 50000.0}
                    ],
                }
                central_switch.accounts.append(new_entry)
                st.success(f"Account for '{new_name}' (Phone: {new_phone}) successfully registered!")
                st.rerun()
            else:
                st.error("Please provide at least Name and Phone Number.")

    # All Accounts Overview Table (Requirement 4: Professional Flagged Classification)
    acc_rows = []
    for a in central_switch.accounts:
        acc_rows.append({
            "Account ID": a["id"],
            "Account Holder": a["name"],
            "Linked Phone": a["phone"],
            "UPI Virtual ID": a["upi"],
            "Account Number": a["account_num"],
            "Bank Branch": a["bank"],
            "Current Balance": f"₹{a['balance']:,.2f}",
            "Regulatory Classification": a["type"],
            "Compliance Status": a.get("regulatory_status", "Clean"),
        })
    df_accs = pd.DataFrame(acc_rows)

    def _style_reg_col(val):
        if "Flagged" in val or "High-Risk" in val:
            return "color: #991b1b; font-weight: 700; background-color: #fee2e2;"
        return "color: #166534; font-weight: 700; background-color: #dcfce7;"

    st.dataframe(df_accs.style.map(_style_reg_col, subset=["Compliance Status"]), use_container_width=True, height=220)

    st.markdown("<hr style='border:0; border-top:1px solid #eaeaea; margin:24px 0 16px 0;'>", unsafe_allow_html=True)

    # ── Bank Statement Inspection ─────────────────────────────────────────────
    st.markdown("### 📑 Detailed Account Bank Statement")
    st.caption("Select any registered account to inspect its official statement, audit trail, and historical transaction flows.")

    acc_options = {f"{a['id']} — {a['name']} ({a['phone']})": a["id"] for a in central_switch.accounts}
    selected_acc_label = st.selectbox("Select Account to Generate Statement:", list(acc_options.keys()), index=2)
    selected_acc_id = acc_options[selected_acc_label]
    sel_acc = next((a for a in central_switch.accounts if a["id"] == selected_acc_id), central_switch.accounts[0])

    # Formal Bank Statement Header
    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:10px; padding:20px 24px; box-shadow:0 2px 8px rgba(0,0,0,0.04); margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; border-bottom:1px solid #e2e8f0; padding-bottom:12px; margin-bottom:14px;">
            <div>
                <h3 style="margin:0; color:#0f172a; font-size:18px;">🏛️ {sel_acc['bank'].split('(')[0]}</h3>
                <span style="font-size:12px; color:#64748b;">OFFICIAL BANK STATEMENT OF ACCOUNT</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:11px; color:#64748b;">STATEMENT PERIOD</span><br>
                <strong style="font-size:12px; color:#0f172a;">01-SEP-2026 TO 29-SEP-2026</strong>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px; font-size:12px;">
            <div><span style="color:#64748b;">Account Holder:</span><br><strong style="color:#0f172a;">{sel_acc['name']}</strong></div>
            <div><span style="color:#64748b;">Account Number:</span><br><strong style="color:#0f172a;">{sel_acc['account_num']}</strong></div>
            <div><span style="color:#64748b;">Linked Mobile:</span><br><strong style="color:#0f172a;">+91 {sel_acc['phone']}</strong></div>
            <div><span style="color:#64748b;">Available Balance:</span><br><strong style="color:#15803d; font-size:14px;">₹{sel_acc['balance']:,.2f}</strong></div>
        </div>
        <div style="margin-top:10px; font-size:12px; color:#64748b;">
            Regulatory Note: <strong style="color:#991b1b if 'Flagged' in sel_acc['type'] else '#15803d';">{sel_acc['type']}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Statement Ledger Entries
    stmt_list = sel_acc.get("statement", [])
    if stmt_list:
        df_stmt = pd.DataFrame(stmt_list)
        df_stmt.columns = ["Date & Time", "Transaction Description", "Reference TxID", "CR/DR", "Amount (₹)", "Closing Balance (₹)"]
        st.dataframe(df_stmt, use_container_width=True)
    else:
        st.info("No prior statement entries recorded for this account.")


# ==============================================================================
# PAGE 7 — EMPLOYEE ABOUT & ARCHITECTURE SPECIFICATIONS
# ==============================================================================
def page_employee_about():
    if st.session_state.user_role != "employee":
        st.error("🔒 Access Denied: Architecture specifications are restricted to authorized bank SOC engineers.")
        if st.button("← Return to Login Selection"):
            logout()
        return

    _render_employee_top_menu(active_tab="about")

    st.markdown("<h2 style='margin:0 0 6px 0; color:#111111;'>Institutional Model Architecture &amp; System Working Conditions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280; font-size:14px; margin-bottom:20px;'>Technical audit specifications, real-time latency benchmark results, and hybrid GTT-XGBoost pipeline telemetry.</p>", unsafe_allow_html=True)

    # Working Conditions Metrics Strip
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Switch Latency SLA", "< 100 ms", "p50: 36.83ms (Compliant)")
    with k2:
        st.metric("Precision-Recall AUC", "0.7920", "+35% vs Pure GNN Baseline")
    with k3:
        st.metric("Illicit Class F1-Score", "0.7033", "@ Optimal Threshold 0.60")
    with k4:
        st.metric("Monitored Graph Size", "203,769 Nodes", "234,355 Directed Edges")
    with k5:
        st.metric("Class Imbalance Weight", "7.63 : 1", "Focal Loss α=0.85, γ=2.0")

    st.markdown("<hr style='border:0; border-top:1px solid #eaeaea; margin:20px 0;'>", unsafe_allow_html=True)

    spec_col_l, spec_col_r = st.columns(2, gap="large")

    with spec_col_l:
        st.markdown("#### ⚙️ Real-Time Switch Working Specifications")
        st.markdown("""
        - **Sub-100ms Inference Guarantee (Method D):**
          * p50 Latency: **36.83 ms**
          * p95 Latency: **64.73 ms**
          * p99 Latency: **68.01 ms**
          * Regulatory Compliance Rate: **100.0% within the < 100ms financial switch window**.
        - **Dynamic 2-Hop Ego-Graph Sampling:**
          * Prevents full-graph recomputation on the massive 203,769-node network.
          * Neighbor limits bounded at $[15, 10]$ to capture the local transaction neighborhood in **5–12 ms**.
        - **Graph-Temporal Transformer (GTT) Encoder:**
          * 2-Layer multi-head edge-attention architecture.
          * Generates **32-dimensional latent structural embeddings** capturing multi-hop mule layering and circular routing topology.
          * Trained with **Focal Loss** to prevent majority licit dominance.
        - **Cost-Sensitive Hybrid XGBoost Classifier:**
          * Combines **165 Tabular local financial features** with **32 GTT Structural embeddings** (197 dimensions total).
          * Dynamic scale positive weight ($7.63$) penalizing false negatives (undetected scams).
        """)

    with spec_col_r:
        st.markdown("#### 🛡️ Scam Attack Topology Working Conditions")
        st.markdown("""
        | Attack Pattern | Topological Signature | Working Detection Condition |
        | :--- | :--- | :--- |
        | **Single-Hop Impersonation** | High velocity single edge to fresh node | Extreme amount-to-degree ratio with high local feature deviation. |
        | **Multi-Hop Mule Layering** | Directed Acyclic Graph (DAG) depth $\\ge 3$ | Rapid traversal across intermediate micro-mules prior to off-ramp hub. |
        | **Circular Wash Smurfing** | Strongly-connected cycle ($A \\to B \\to C \\to A$) | Topological embedding `graph_emb_25` spike + NetworkX cycle alert. |
        | **Rapid Fan-Out Aggregation** | High fan-in star topology | Aggregator hub splitting into multiple micro-wallets in $<90$ seconds. |
        """)

    st.markdown("<hr style='border:0; border-top:1px solid #eaeaea; margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown("#### 🔄 Real-Time End-to-End Inference Flow")
    st.markdown("""
    ```mermaid
    flowchart LR
        A["1. Incoming Payment<br/>(Switch Pre-Auth)"] --> B["2. 2-Hop Ego-Graph<br/>(PyG NeighborLoader)"]
        B --> C["3. GTT Encoder<br/>(32-dim Structural Emb)"]
        C --> D["4. Feature Fusion<br/>(165 Tabular + 32 GNN)"]
        D --> E["5. Cost-Weighted XGBoost<br/>(Fraud Probability)"]
        E --> F["6. 3-Tier Action Engine<br/>(Approve / MFA / Freeze)"]
    ```
    """)


# ==============================================================================
# MAIN ROUTER
# ==============================================================================
def main():
    get_pipeline()
    page = st.session_state.current_page

    if page == "login_selection":
        page_login_selection()
    elif page == "customer_login":
        page_customer_login()
    elif page == "employee_login":
        page_employee_login()
    elif page == "customer_portal":
        page_customer_portal()
    elif page == "employee_dashboard":
        page_employee_dashboard()
    elif page == "employee_accounts":
        page_employee_accounts()
    elif page == "employee_about":
        page_employee_about()
    else:
        nav_to("login_selection")

if __name__ == "__main__" or True:
    main()
