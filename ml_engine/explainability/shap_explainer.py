import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

# Headless backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class FraudSHAPExplainer:
    """
    Production-grade Explainable AI (XAI) engine using TreeSHAP to attribute
    predictions of the Hybrid Tabular-GNN XGBoost model.

    Features:
    - Attribution across both raw transaction signals and GNN structural embeddings.
    - Global summary bar and beeswarm visualizations.
    - Domain feature importance breakdown (Tabular vs Topological Graph Embeddings).
    - Sub-20ms low-latency single transaction local attribution.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        root_dir = Path(__file__).resolve().parent.parent.parent
        self.model_path = (
            Path(model_path) if model_path else (root_dir / "artifacts" / "saved_models" / "hybrid_xgboost_best.json")
        )
        self.config_path = (
            Path(config_path) if config_path else (root_dir / "artifacts" / "saved_models" / "model_config.json")
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at: {self.model_path}")

        # Load XGBoost model
        logger.info(f"Loading trained XGBoost model from {self.model_path}...")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))

        # Load configuration
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            self.feature_names = self.config.get("feature_names", [f"feat_{i}" for i in range(197)])
            self.optimal_threshold = self.config.get("optimal_threshold", 0.50)
        else:
            self.config = {}
            self.feature_names = [f"feat_{i}" for i in range(197)]
            self.optimal_threshold = 0.50

        # Initialize TreeExplainer
        logger.info("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("FraudSHAPExplainer successfully initialized.")

    def compute_and_save_global_explanations(
        self,
        X_sample: np.ndarray,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes global SHAP attributions over a representative background sample,
        generates summary bar and beeswarm plots, and computes the domain contribution
        of GNN structural embeddings vs raw tabular attributes.

        Args:
            X_sample: Background sample of shape [N, 197].
            output_dir: Destination directory for explanation artifacts.

        Returns:
            Dict containing domain breakdown metrics and ranking summary.
        """
        out_dir = Path(output_dir) if output_dir else Path("artifacts/logs")
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Computing global SHAP values over {len(X_sample):,} sample records...")
        shap_explanation = self.explainer(X_sample)

        # Handle 2D vs 3D SHAP values (binary classification output)
        if len(shap_explanation.values.shape) == 3:
            shap_matrix = shap_explanation.values[:, :, 1]
        else:
            shap_matrix = shap_explanation.values

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        total_shap_mass = np.sum(mean_abs_shap)

        # -------------------------------------------------------------
        # 1. Domain Feature Importance Breakdown
        # -------------------------------------------------------------
        tabular_mask = np.array([name.startswith("raw_feat_") for name in self.feature_names])
        graph_mask = np.array([name.startswith("graph_emb_") for name in self.feature_names])

        tabular_importance = np.sum(mean_abs_shap[tabular_mask])
        graph_importance = np.sum(mean_abs_shap[graph_mask])

        pct_tabular = (tabular_importance / max(1e-7, total_shap_mass)) * 100
        pct_graph = (graph_importance / max(1e-7, total_shap_mass)) * 100

        logger.info("=" * 65)
        logger.info("SHAP DOMAIN CONTRIBUTION BREAKDOWN")
        logger.info("=" * 65)
        logger.info(f"  • Tabular Features (165 raw signals):       {tabular_importance:.4f} ({pct_tabular:.2f}%)")
        logger.info(f"  • Topological GNN Embeddings (32 dims):     {graph_importance:.4f} ({pct_graph:.2f}%)")
        logger.info(f"  • Total Explanatory Power Mass:             {total_shap_mass:.4f}")
        logger.info("=" * 65)

        # -------------------------------------------------------------
        # 2. Save Feature Importance Ranking CSV
        # -------------------------------------------------------------
        ranking_data = []
        for rank_idx, feat_idx in enumerate(np.argsort(mean_abs_shap)[::-1]):
            feat_name = self.feature_names[feat_idx]
            domain = "Structural_Embedding" if feat_name.startswith("graph_emb_") else "Raw_Tabular"
            imp = float(mean_abs_shap[feat_idx])
            rel_pct = float((imp / max(1e-7, total_shap_mass)) * 100)
            ranking_data.append({
                "rank": rank_idx + 1,
                "feature_name": feat_name,
                "domain": domain,
                "mean_abs_shap": imp,
                "relative_importance_pct": rel_pct,
            })

        df_ranking = pd.DataFrame(ranking_data)
        csv_path = out_dir / "feature_importance_ranking.csv"
        df_ranking.to_csv(csv_path, index=False)
        logger.info(f"Saved full feature importance rankings to: {csv_path}")

        # -------------------------------------------------------------
        # 3. Global Summary Bar Plot
        # -------------------------------------------------------------
        bar_plot_path = out_dir / "shap_bar_plot.png"
        logger.info(f"Generating Global SHAP Bar Plot -> {bar_plot_path}...")
        plt.figure(figsize=(10, 8), dpi=300)
        shap.summary_plot(
            shap_matrix,
            X_sample,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=20,
            show=False,
        )
        plt.title("Top 20 Most Impactful Features (Global SHAP Mean |Value|)", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(bar_plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # 4. Global Beeswarm Plot
        # -------------------------------------------------------------
        beeswarm_path = out_dir / "shap_beeswarm_plot.png"
        logger.info(f"Generating Global SHAP Beeswarm Plot -> {beeswarm_path}...")
        plt.figure(figsize=(10, 8), dpi=300)
        shap.summary_plot(
            shap_matrix,
            X_sample,
            feature_names=self.feature_names,
            max_display=20,
            show=False,
        )
        plt.title("SHAP Beeswarm Plot (Directional Fraud Push)", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info("Global SHAP visualizations successfully generated.")

        return {
            "tabular_importance": float(tabular_importance),
            "graph_importance": float(graph_importance),
            "pct_tabular": float(pct_tabular),
            "pct_graph": float(pct_graph),
            "top_5_features": ranking_data[:5],
            "bar_plot_file": str(bar_plot_path),
            "beeswarm_plot_file": str(beeswarm_path),
            "ranking_csv": str(csv_path),
        }

    def explain_single_transaction(self, x_vector: np.ndarray) -> Dict[str, Any]:
        """
        Executes ultra-low latency (<20ms) local attribution for an incoming transaction.

        Args:
            x_vector: 197-dimensional array representing the transaction.

        Returns:
            Dict containing predicted fraud probability, risk classification,
            and top 5 positive and negative contributing feature deltas.
        """
        x_in = np.array(x_vector, dtype=np.float32)
        if x_in.ndim == 1:
            x_in = x_in.reshape(1, -1)

        # Probabilistic prediction
        prob_fraud = float(self.model.predict_proba(x_in)[0, 1])

        # Compute SHAP values
        explanation = self.explainer(x_in)
        if len(explanation.values.shape) == 3:
            shap_values = explanation.values[0, :, 1]
            base_val = float(explanation.base_values[0, 1])
        else:
            shap_values = explanation.values[0, :]
            base_val = float(explanation.base_values[0])

        # Risk classification against policy threshold
        if prob_fraud >= self.optimal_threshold:
            risk_level = "HIGH (FRAUD ALERT / AUTO-FREEZE)"
        elif prob_fraud >= self.optimal_threshold * 0.7:
            risk_level = "MEDIUM (ELEVATED MONITORING)"
        else:
            risk_level = "LOW (LEGITIMATE)"

        # Sort feature impacts
        sorted_indices = np.argsort(shap_values)

        # Top 5 pushing towards fraud (positive SHAP)
        top_positive_idx = sorted_indices[-5:][::-1]
        top_positive = [
            {
                "feature": self.feature_names[i],
                "domain": "GNN_Embedding" if self.feature_names[i].startswith("graph_emb_") else "Tabular",
                "shap_delta": float(shap_values[i]),
                "feature_value": float(x_in[0, i]),
            }
            for i in top_positive_idx
            if shap_values[i] > 0
        ]

        # Top 5 pushing away from fraud (negative SHAP)
        top_negative_idx = sorted_indices[:5]
        top_negative = [
            {
                "feature": self.feature_names[i],
                "domain": "GNN_Embedding" if self.feature_names[i].startswith("graph_emb_") else "Tabular",
                "shap_delta": float(shap_values[i]),
                "feature_value": float(x_in[0, i]),
            }
            for i in top_negative_idx
            if shap_values[i] < 0
        ]

        return {
            "fraud_probability": round(prob_fraud, 4),
            "optimal_threshold": self.optimal_threshold,
            "risk_level": risk_level,
            "base_value": round(base_val, 4),
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
        }
