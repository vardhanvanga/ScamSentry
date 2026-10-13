"""
Unit and Integration Tests for Phase 5 Enterprise Gateway & Compliance.
"""
import unittest
import uuid
from pathlib import Path
from pydantic import ValidationError

from backend.app.engine.compliance import DomainFeatureMapper, RegulatoryAuditLogger
from backend.app.engine.action_engine import PolicyResponse
from backend.app.main import TransactionPayload, SwitchDecisionResponse, ExplainabilityEntry


class TestCompliancePillar(unittest.TestCase):
    """Verifies RBI FREE-AI & DPDP compliance mappers and audit logging."""

    def setUp(self):
        self.mapper = DomainFeatureMapper()
        self.test_log_path = Path("artifacts/logs/test_regulatory_audit.jsonl")
        self.logger = RegulatoryAuditLogger(log_path=self.test_log_path)

    def tearDown(self):
        if self.test_log_path.exists():
            try:
                self.test_log_path.unlink()
            except Exception:
                pass

    def test_tabular_feature_mapping(self):
        # Velocity burst (0-9)
        domain = self.mapper.map_feature_to_domain("raw_feat_4")
        self.assertIn("velocity burst", domain.lower())

        # Structuring / Smurfing (10-29)
        domain = self.mapper.map_feature_to_domain("raw_feat_15")
        self.assertIn("structuring", domain.lower())

        # Dormant reactivation (30-49)
        domain = self.mapper.map_feature_to_domain("raw_feat_35")
        self.assertIn("dormant", domain.lower())

    def test_graph_embedding_mapping(self):
        # GNN embedding 0 (absolute 165) -> peel chain
        domain = self.mapper.map_feature_to_domain("graph_emb_0")
        self.assertIn("peel chain", domain.lower())

        # GNN embedding 12 (absolute 177) -> mule hub
        domain = self.mapper.map_feature_to_domain("graph_emb_12")
        self.assertIn("mule hub", domain.lower())

    def test_explainability_summary_formatting(self):
        dummy_shap = {
            "top_positive_features": [
                {"feature": "raw_feat_5", "domain": "Tabular", "shap_delta": 0.42, "feature_value": 3.14}
            ],
            "top_negative_features": [
                {"feature": "raw_feat_85", "domain": "Tabular", "shap_delta": -0.15, "feature_value": 0.05}
            ]
        }
        summary = self.mapper.format_explainability_summary(dummy_shap)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["direction"], "RISK_ELEVATING")
        self.assertEqual(summary[1]["direction"], "RISK_MITIGATING")
        self.assertEqual(summary[0]["feature_name"], "raw_feat_5")

    def test_regulatory_audit_logger(self):
        trace_id = str(uuid.uuid4())
        payload = {"tx_id": "TEST_TX_001", "source_account_id": "ACC_999", "amount": 50000.0}
        decision = PolicyResponse(
            tx_id="TEST_TX_001",
            risk_score=0.92,
            risk_tier="HIGH_RISK",
            recommended_action="FLAG_HIGH_RISK",
            reason_codes=["High structural similarity to known money mule hub"],
            latency_ms=45.2,
            switch_compliance=True,
        )
        shap_summary = [{"feature_name": "graph_emb_12", "domain_label": "Money mule hub", "attribution_weight": 0.5, "feature_value": 1.0, "direction": "RISK_ELEVATING"}]

        # Write log
        self.logger.log_flagged_transaction(trace_id, payload, decision, shap_summary)

        # Retrieve log
        trail = self.logger.get_recent_audit_trail(n=5)
        self.assertEqual(len(trail), 1)
        record = trail[0]
        self.assertEqual(record["trace_id"], trace_id)
        self.assertEqual(record["action"], "FLAG_HIGH_RISK")
        self.assertIn("RBI FREE-AI", record["regulatory_note"])


class TestGatewaySchemas(unittest.TestCase):
    """Verifies Pydantic V2 payload and response schema contracts."""

    def test_valid_transaction_payload(self):
        data = {
            "tx_id": "TXN_VALID_123",
            "source_account_id": "ACC_101",
            "target_node_idx": 50,
            "amount": 2500.0,
            "timestamp": 1727000000,
        }
        payload = TransactionPayload(**data)
        self.assertEqual(payload.tx_id, "TXN_VALID_123")
        self.assertEqual(payload.amount, 2500.0)

    def test_invalid_transaction_payload(self):
        # Invalid amount (<= 0)
        with self.assertRaises(ValidationError):
            TransactionPayload(
                tx_id="TXN_INVALID",
                source_account_id="ACC_101",
                target_node_idx=50,
                amount=-10.0,
                timestamp=1727000000,
            )

        # Negative node index
        with self.assertRaises(ValidationError):
            TransactionPayload(
                tx_id="TXN_INVALID",
                source_account_id="ACC_101",
                target_node_idx=-5,
                amount=100.0,
                timestamp=1727000000,
            )


if __name__ == "__main__":
    unittest.main()
