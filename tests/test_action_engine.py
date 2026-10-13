import unittest
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.engine.action_engine import (
    RiskTriagingEngine,
    TransactionRequest,
    PolicyResponse,
)


class TestRiskTriagingEngine(unittest.TestCase):
    """Unit tests for regulatory 3-tier risk-triaging policy engine and SLA checks."""

    def setUp(self):
        self.engine = RiskTriagingEngine(low_threshold=0.40, high_threshold=0.85)

    def test_01_boundary_conditions(self):
        """1. Assert exact boundary tier mappings for 0.39, 0.40, 0.84, and 0.85."""
        # 0.39 -> LOW_RISK / APPROVE
        resp_039 = self.engine.evaluate("tx_039", 0.39, latency_ms=12.0)
        self.assertEqual(resp_039.risk_tier, "LOW_RISK")
        self.assertEqual(resp_039.recommended_action, "APPROVE")
        self.assertNotIn("tx_039", self.engine.flagged_syndicate_nodes)

        # 0.40 -> MEDIUM_RISK / STEP_UP_MFA
        resp_040 = self.engine.evaluate("tx_040", 0.40, latency_ms=15.0)
        self.assertEqual(resp_040.risk_tier, "MEDIUM_RISK")
        self.assertEqual(resp_040.recommended_action, "STEP_UP_MFA")
        self.assertNotIn("tx_040", self.engine.flagged_syndicate_nodes)

        # 0.84 -> MEDIUM_RISK / STEP_UP_MFA
        resp_084 = self.engine.evaluate("tx_084", 0.84, latency_ms=18.0)
        self.assertEqual(resp_084.risk_tier, "MEDIUM_RISK")
        self.assertEqual(resp_084.recommended_action, "STEP_UP_MFA")
        self.assertNotIn("tx_084", self.engine.flagged_syndicate_nodes)

        # 0.85 -> HIGH_RISK / FLAG_HIGH_RISK
        resp_085 = self.engine.evaluate("tx_085", 0.85, latency_ms=22.0)
        self.assertEqual(resp_085.risk_tier, "HIGH_RISK")
        self.assertEqual(resp_085.recommended_action, "FLAG_HIGH_RISK")
        self.assertIn("tx_085", self.engine.flagged_syndicate_nodes)

    def test_02_switch_latency_compliance_flag(self):
        """2. Assert switch_compliance is True under 100ms and False at/above 100ms."""
        # Under 100ms -> compliant
        fast_resp = self.engine.evaluate("tx_fast", 0.15, latency_ms=45.2)
        self.assertTrue(fast_resp.switch_compliance)
        self.assertEqual(fast_resp.latency_ms, 45.2)

        # At/Above 100ms -> non-compliant
        slow_resp = self.engine.evaluate("tx_slow", 0.15, latency_ms=105.8)
        self.assertFalse(slow_resp.switch_compliance)
        self.assertEqual(slow_resp.latency_ms, 105.8)

        # Exact boundary 100.0ms -> non-compliant (< 100.0)
        exact_resp = self.engine.evaluate("tx_exact", 0.15, latency_ms=100.0)
        self.assertFalse(exact_resp.switch_compliance)

    def test_03_transaction_request_dataclass(self):
        """3. Assert TransactionRequest instantiation and attribute integrity."""
        req = TransactionRequest(
            tx_id="tx_test_100",
            source_node=100,
            amount=2.54,
            timestamp=36,
            features=[0.1] * 165,
        )
        self.assertEqual(req.tx_id, "tx_test_100")
        self.assertEqual(req.source_node, 100)
        self.assertEqual(req.amount, 2.54)
        self.assertEqual(len(req.features), 165)


if __name__ == "__main__":
    unittest.main(verbosity=2)
