"""
compliance.py — ScamSentry STAR-Fraud Engine · Pillar 2: Regulatory Compliance Module
======================================================================================

Provides two production-grade classes:

    DomainFeatureMapper
        Maps raw SHAP feature names (raw_feat_N / graph_emb_N) to human-readable
        banking-domain concepts and formats per-transaction explainability summaries
        suitable for adverse-action notices.

    RegulatoryAuditLogger
        Writes immutable, append-only JSONL audit records for flagged transactions,
        conforming to RBI FREE-AI and DPDP audit standards.

Feature-vector layout (197 dims):
    Indices 0–164   → tabular features  (raw_feat_0 … raw_feat_164)
    Indices 165–196 → graph embeddings  (graph_emb_0 … graph_emb_31)

Usage:
    from backend.app.engine.compliance import DomainFeatureMapper, RegulatoryAuditLogger
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Project root — resolve relative to this file's location
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[3]  # …/Graph Sentinal


# ---------------------------------------------------------------------------
# Forward declaration guard — avoids circular imports at runtime.
# PolicyResponse is imported only for type-checking.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from backend.app.engine.policy import PolicyResponse  # noqa: F401


# ===========================================================================
# 1. DomainFeatureMapper
# ===========================================================================

class DomainFeatureMapper:
    """Translates SHAP feature identifiers into human-readable banking-domain labels.

    The 197-dim hybrid feature vector is split into two logical groups:

    Tabular features (raw_feat_0 … raw_feat_164)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Each index range maps to a named financial-crime typology so that
    adverse-action notices are meaningful to compliance analysts.

    Graph-embedding features (graph_emb_0 … graph_emb_31)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Dimensions of the GNN embedding space are grouped by conceptual role
    (peel-chain depth, mule-hub similarity, fan-in/fan-out ratio).

    Examples
    --------
    >>> mapper = DomainFeatureMapper()
    >>> mapper.map_feature_to_domain('raw_feat_5')
    'Transaction velocity burst — rapid high-frequency outbound activity'
    >>> mapper.map_feature_to_domain('graph_emb_12')
    'Multi-hop layered peel chain pattern detected'
    """

    # ------------------------------------------------------------------
    # Tabular domain map — (start_index_inclusive, end_index_inclusive)
    # ------------------------------------------------------------------
    TABULAR_DOMAIN_MAP: dict[tuple[int, int], str] = {
        (0,   9):   "Transaction velocity burst — rapid high-frequency outbound activity",
        (10,  29):  "Structuring / Smurfing indicator — amounts systematically near reporting thresholds",
        (30,  49):  "Dormant account reactivation — sudden activity after prolonged inactivity",
        (50,  79):  "Fan-out aggregation pattern — disproportionate number of unique counterparties",
        (80,  109): "Temporal clustering — irregular time-of-day / day-of-week transaction cadence",
        (110, 139): "Cross-border layering signal — unusual cross-jurisdictional routing",
        (140, 164): "Balance manipulation indicator — rapid fund accumulation and immediate dispersal",
    }

    # ------------------------------------------------------------------
    # Graph-embedding domain map — absolute feature-vector indices
    # ------------------------------------------------------------------
    GRAPH_DOMAIN_MAP: dict[tuple[int, int], str] = {
        (165, 174): "Multi-hop layered peel chain pattern detected",
        (175, 184): "High structural similarity to known money mule hub",
        (185, 196): "Disproportionate fan-in/fan-out flow ratio",
    }

    # Compiled patterns for fast parsing
    _RAW_PATTERN: re.Pattern[str] = re.compile(r"^raw_feat_(\d+)$")
    _GRAPH_PATTERN: re.Pattern[str] = re.compile(r"^graph_emb_(\d+)$")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_feature_to_domain(self, feature_name: str) -> str:
        """Return the human-readable banking-domain label for a SHAP feature name.

        Parameters
        ----------
        feature_name:
            One of ``raw_feat_N`` (N ∈ 0–164) or ``graph_emb_N`` (N ∈ 0–31).

        Returns
        -------
        str
            Descriptive domain string, or ``'General transaction anomaly'`` when
            the feature name does not match any known range.
        """
        raw_match = self._RAW_PATTERN.match(feature_name)
        if raw_match:
            idx = int(raw_match.group(1))
            for (lo, hi), label in self.TABULAR_DOMAIN_MAP.items():
                if lo <= idx <= hi:
                    return label
            return "General transaction anomaly"

        graph_match = self._GRAPH_PATTERN.match(feature_name)
        if graph_match:
            # graph_emb_N → absolute vector index = 165 + N
            relative_idx = int(graph_match.group(1))
            absolute_idx = 165 + relative_idx
            for (lo, hi), label in self.GRAPH_DOMAIN_MAP.items():
                if lo <= absolute_idx <= hi:
                    return label
            return "General transaction anomaly"

        return "General transaction anomaly"

    def format_explainability_summary(self, shap_result: dict) -> list[dict]:
        """Build a structured explainability summary from a SHAP result dict.

        Combines ``top_positive_features`` and ``top_negative_features`` from
        the output of ``FraudSHAPExplainer.explain_single_transaction()``.

        Parameters
        ----------
        shap_result:
            Dict produced by ``FraudSHAPExplainer.explain_single_transaction()``.
            Expected to contain:

            - ``top_positive_features``: list of dicts with keys
              ``feature_name``, ``shap_delta``, ``feature_value``
            - ``top_negative_features``: list of dicts with the same keys

        Returns
        -------
        list[dict]
            Each entry contains:

            ================= ================================================
            Key               Description
            ================= ================================================
            feature_name      Original SHAP feature identifier
            domain_label      Human-readable banking-domain concept
            attribution_weight Absolute SHAP delta value (float)
            feature_value     Raw feature value at inference time
            direction         ``'RISK_ELEVATING'`` or ``'RISK_MITIGATING'``
            ================= ================================================

        Notes
        -----
        The returned list is ordered: positive contributors first, then
        negative contributors, preserving original ranking within each group.
        """
        summary: list[dict] = []

        positive_features: list[dict] = shap_result.get("top_positive_features", [])
        negative_features: list[dict] = shap_result.get("top_negative_features", [])

        for feat in positive_features:
            feature_name: str = feat.get("feature_name") or feat.get("feature", "")
            shap_delta: float = float(feat.get("shap_delta", 0.0))
            summary.append(
                {
                    "feature_name": feature_name,
                    "domain_label": self.map_feature_to_domain(feature_name),
                    "attribution_weight": round(abs(shap_delta), 6),
                    "feature_value": feat.get("feature_value"),
                    "direction": "RISK_ELEVATING" if shap_delta > 0 else "RISK_MITIGATING",
                }
            )

        for feat in negative_features:
            feature_name = feat.get("feature_name") or feat.get("feature", "")
            shap_delta = float(feat.get("shap_delta", 0.0))
            summary.append(
                {
                    "feature_name": feature_name,
                    "domain_label": self.map_feature_to_domain(feature_name),
                    "attribution_weight": round(abs(shap_delta), 6),
                    "feature_value": feat.get("feature_value"),
                    "direction": "RISK_ELEVATING" if shap_delta > 0 else "RISK_MITIGATING",
                }
            )

        return summary


# ===========================================================================
# 2. RegulatoryAuditLogger
# ===========================================================================

# Actions that trigger an immutable audit record (regulatory requirement)
_AUDIT_TRIGGER_ACTIONS: frozenset[str] = frozenset(
    {"FLAG_HIGH_RISK", "STEP_UP_MFA"}
)

_REGULATORY_NOTE: str = (
    "Adverse action notice generated per RBI FREE-AI and DPDP audit standards. "
    "Immutable record."
)


class RegulatoryAuditLogger:
    """Append-only JSONL audit trail writer for flagged transactions.

    Conforms to RBI FREE-AI and DPDP audit requirements by writing one
    JSON-line per flagged event to a persistent file.  Records are never
    modified or deleted after writing.

    Only transactions whose recommended action is ``FLAG_HIGH_RISK`` or
    ``STEP_UP_MFA`` are persisted; all other actions are silently skipped.

    Parameters
    ----------
    log_path:
        Absolute path to the JSONL audit file.  Defaults to
        ``<ROOT_DIR>/artifacts/logs/regulatory_audit_trail.jsonl``.
        Parent directories are created automatically on first use.

    Examples
    --------
    >>> logger = RegulatoryAuditLogger()
    >>> logger.log_flagged_transaction(
    ...     trace_id="trace-abc123",
    ...     payload={"source_account_id": "ACC001", "amount": 49900},
    ...     decision=policy_response,
    ...     shap_summary=[...],
    ... )
    """

    DEFAULT_LOG_PATH: Path = ROOT_DIR / "artifacts" / "logs" / "regulatory_audit_trail.jsonl"

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path: Path = Path(log_path) if log_path is not None else self.DEFAULT_LOG_PATH
        # Ensure parent directories exist (idempotent)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_flagged_transaction(
        self,
        trace_id: str,
        payload: dict,
        decision: "PolicyResponse",
        shap_summary: list[dict],
    ) -> None:
        """Append one audit record to the JSONL trail if the action warrants it.

        Parameters
        ----------
        trace_id:
            Distributed-trace correlation identifier for the request.
        payload:
            Original inference request payload dict.  Expected to contain at
            minimum ``source_account_id`` and ``amount``.
        decision:
            ``PolicyResponse`` dataclass produced by the policy engine.
            Must expose: ``tx_id``, ``risk_score``, ``risk_tier``,
            ``recommended_action``, ``reason_codes``, ``latency_ms``,
            ``switch_compliance``.
        shap_summary:
            Structured explainability list from
            ``DomainFeatureMapper.format_explainability_summary()``.

        Notes
        -----
        - Records are written atomically as a single ``json.dumps`` + newline.
        - The file is opened in *append* mode on every call to avoid keeping
          file handles open between requests (safe for multi-process deploys).
        - Non-triggering actions (e.g. ``ALLOW``, ``SOFT_DECLINE``) are
          silently skipped without raising.
        """
        action: str = str(decision.recommended_action)
        if action not in _AUDIT_TRIGGER_ACTIONS:
            return

        record: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "tx_id": decision.tx_id,
            "source_account_id": payload.get("source_account_id"),
            "amount": payload.get("amount"),
            "risk_score": decision.risk_score,
            "risk_tier": str(decision.risk_tier),
            "action": action,
            "latency_ms": decision.latency_ms,
            "sla_compliant": decision.switch_compliance,
            "reason_codes": decision.reason_codes,
            "explainability_summary": shap_summary,
            "regulatory_note": _REGULATORY_NOTE,
        }

        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_recent_audit_trail(self, n: int = 20) -> list[dict]:
        """Return the last *n* audit records from the JSONL file.

        Parameters
        ----------
        n:
            Maximum number of records to return (default 20).

        Returns
        -------
        list[dict]
            Parsed audit record dicts, ordered oldest-to-newest.
            Returns an empty list if the file does not exist or is empty.
        """
        if not self.log_path.exists():
            return []

        with self.log_path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()

        recent_lines = lines[-n:] if len(lines) > n else lines
        records: list[dict] = []
        for line in recent_lines:
            stripped = line.strip()
            if stripped:
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    # Corrupt line — skip silently to avoid breaking the audit read
                    continue
        return records
