import logging
from dataclasses import dataclass, field
from typing import List, Union, Set, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransactionRequest:
    """Incoming payment switch transaction payload."""
    tx_id: str
    source_node: int
    amount: float
    timestamp: int
    features: Union[List[float], np.ndarray] = field(default_factory=list)


@dataclass
class PolicyResponse:
    """Regulatory-compliant policy evaluation and switch decision."""
    tx_id: str
    risk_score: float
    risk_tier: str  # "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"
    recommended_action: str  # "APPROVE", "STEP_UP_MFA", "FLAG_HIGH_RISK"
    reason_codes: List[str]
    latency_ms: float
    switch_compliance: bool  # True if latency_ms < 100.0


class RiskTriagingEngine:
    """
    Regulatory-compliant 3-tier risk-triaging policy engine for real-time payment switches.

    Tiers:
    - LOW_RISK (< low_threshold): APPROVE (Frictionless baseline flow).
    - MEDIUM_RISK (low_threshold <= risk < high_threshold): STEP_UP_MFA (Dynamic Step-Up OTP/Biometric).
    - HIGH_RISK (>= high_threshold): FLAG_HIGH_RISK (Immediate hold / account freezing).
    """

    def __init__(self, low_threshold: float = 0.40, high_threshold: float = 0.85):
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.flagged_syndicate_nodes: Set[str] = set()

    def evaluate(
        self,
        tx_id: str,
        risk_score: float,
        shap_reasons: Optional[List[str]] = None,
        latency_ms: float = 0.0,
    ) -> PolicyResponse:
        """
        Evaluate transaction risk against multi-tier policy thresholds.

        Args:
            tx_id: Transaction identifier.
            risk_score: Estimated fraud probability [0.0, 1.0].
            shap_reasons: List of human-interpretable SHAP attribution strings.
            latency_ms: Execution latency in milliseconds.

        Returns:
            PolicyResponse with risk tier, recommended action, and switch compliance status.
        """
        reasons = shap_reasons if shap_reasons is not None else []
        risk_val = float(risk_score)

        if risk_val < self.low_threshold:
            risk_tier = "LOW_RISK"
            recommended_action = "APPROVE"
            if not reasons:
                reasons = ["Transaction pattern conforms to regular baseline"]
        elif risk_val < self.high_threshold:
            risk_tier = "MEDIUM_RISK"
            recommended_action = "STEP_UP_MFA"
            if not reasons:
                reasons = ["Borderline risk characteristics detected; dynamic step-up authentication required"]
        else:
            risk_tier = "HIGH_RISK"
            recommended_action = "FLAG_HIGH_RISK"
            self.flagged_syndicate_nodes.add(str(tx_id))
            if not reasons:
                reasons = ["High-confidence illicit syndicate structure detected; payment hold initiated"]

        is_compliant = latency_ms < 100.0

        decision = PolicyResponse(
            tx_id=str(tx_id),
            risk_score=round(risk_val, 4),
            risk_tier=risk_tier,
            recommended_action=recommended_action,
            reason_codes=reasons,
            latency_ms=round(float(latency_ms), 2),
            switch_compliance=is_compliant,
        )

        logger.info(
            f"Policy Decision [Tx: {tx_id}] -> {recommended_action} "
            f"(Score: {risk_val:.4f}, Tier: {risk_tier}, Latency: {latency_ms:.2f}ms, Compliant: {is_compliant})"
        )
        return decision


# Backwards compatibility alias for Phase 1 / Phase 3 callers
class ActionEngine:
    """Legacy compatibility wrapper for RiskTriagingEngine."""

    def __init__(self, risk_threshold: float = 0.75, auto_freeze: bool = True):
        self.risk_threshold = risk_threshold
        self.auto_freeze = auto_freeze
        self.triaging_engine = RiskTriagingEngine(
            low_threshold=risk_threshold * 0.7,
            high_threshold=risk_threshold,
        )

    def evaluate_risk(self, tx_id: str, risk_score: float, subgraph_features: Dict[str, Any] = None) -> Dict[str, Any]:
        resp = self.triaging_engine.evaluate(tx_id, risk_score, latency_ms=1.5)
        action = "FREEZE" if resp.recommended_action == "FLAG_HIGH_RISK" and self.auto_freeze else (
            "ALLOW" if resp.recommended_action == "APPROVE" else "FLAG_MANUAL_REVIEW"
        )
        return {
            "tx_id": resp.tx_id,
            "risk_score": resp.risk_score,
            "threshold": self.risk_threshold,
            "action": action,
            "account_frozen": (action == "FREEZE"),
            "reason": resp.reason_codes[0] if resp.reason_codes else "Evaluated by policy engine",
            "subgraph_features": subgraph_features or {},
        }
