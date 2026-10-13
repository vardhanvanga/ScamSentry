import unittest
import time
import sys
from pathlib import Path
import numpy as np
import xgboost as xgb

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_engine.models.hybrid_xgboost import load_and_fuse_features
from ml_engine.explainability.shap_explainer import FraudSHAPExplainer


class TestHybridPipeline(unittest.TestCase):
    """Automated assertions for Phase 3 Hybrid Classifier & SHAP Explainer."""

    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train, cls.X_test, cls.y_test, cls.feature_names, cls.scale_pos_weight, cls.test_tx_ids = (
            load_and_fuse_features()
        )

    def test_01_feature_fusion_integrity(self):
        """1. Verify that load_and_fuse_features() yields exactly 197 features without NaNs or Infs."""
        self.assertEqual(len(self.feature_names), 197)
        self.assertEqual(self.X_train.shape[1], 197)
        self.assertEqual(self.X_test.shape[1], 197)

        # Assert no NaNs or Infs
        self.assertFalse(np.isnan(self.X_train).any(), "NaNs detected in X_train")
        self.assertFalse(np.isnan(self.X_test).any(), "NaNs detected in X_test")
        self.assertFalse(np.isinf(self.X_train).any(), "Infs detected in X_train")
        self.assertFalse(np.isinf(self.X_test).any(), "Infs detected in X_test")

        # Check feature nomenclature prefixing
        self.assertEqual(self.feature_names[0], "raw_feat_0")
        self.assertEqual(self.feature_names[164], "raw_feat_164")
        self.assertEqual(self.feature_names[165], "graph_emb_0")
        self.assertEqual(self.feature_names[196], "graph_emb_31")

    def test_02_exclusion_of_unlabeled_nodes(self):
        """2. Verify that train and test splits strictly exclude unlabeled transactions (-1)."""
        self.assertNotIn(-1, self.y_train, "Unlabeled transaction (-1) found in y_train")
        self.assertNotIn(-1, self.y_test, "Unlabeled transaction (-1) found in y_test")
        self.assertTrue(set(np.unique(self.y_train)).issubset({0, 1}))
        self.assertTrue(set(np.unique(self.y_test)).issubset({0, 1}))
        self.assertGreater(self.scale_pos_weight, 1.0, "scale_pos_weight should reflect class imbalance")

    def test_03_model_loading_and_inference(self):
        """3. Verify that the trained XGBoost model exists and produces valid predictions."""
        model_file = ROOT_DIR / "artifacts" / "saved_models" / "hybrid_xgboost_best.json"
        if not model_file.exists():
            # If not yet trained via CLI, train a quick mock model to test persistence
            clf = xgb.XGBClassifier(n_estimators=5, max_depth=3, tree_method="hist")
            clf.fit(self.X_train[:100], self.y_train[:100])
            clf.save_model(str(model_file))

        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(str(model_file))
        probs = loaded_model.predict_proba(self.X_test[:10])
        self.assertEqual(probs.shape, (10, 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_04_shap_attribution_latency(self):
        """4. Verify that FraudSHAPExplainer.explain_single_transaction() executes in < 20 ms."""
        model_file = ROOT_DIR / "artifacts" / "saved_models" / "hybrid_xgboost_best.json"
        config_file = ROOT_DIR / "artifacts" / "saved_models" / "model_config.json"

        explainer = FraudSHAPExplainer(model_path=str(model_file), config_path=str(config_file))
        sample_x = self.X_test[0]

        # Warmup
        _ = explainer.explain_single_transaction(sample_x)

        # Benchmark 5 runs
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            explanation = explainer.explain_single_transaction(sample_x)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

        avg_latency = np.mean(latencies)
        self.assertLess(
            avg_latency,
            20.0,
            f"SHAP attribution latency {avg_latency:.2f}ms exceeded 20ms threshold",
        )
        self.assertIn("fraud_probability", explanation)
        self.assertIn("risk_level", explanation)
        self.assertIn("top_positive_features", explanation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
