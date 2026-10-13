import time
import logging
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import torch
import xgboost as xgb
from torch_geometric.utils import k_hop_subgraph

from backend.app.engine.action_engine import RiskTriagingEngine, PolicyResponse
from ml_engine.models.graph_transformer import GraphTemporalTransformer
from ml_engine.explainability.shap_explainer import FraudSHAPExplainer

logger = logging.getLogger(__name__)


class RealTimeInferencePipeline:
    """
    Production-grade Real-Time Streaming Ingestion and Low-Latency Graph Inference Engine.

    Features:
    - 2-hop local ego-graph dynamic extraction (avoids full-graph re-computation).
    - Subgraph Forward pass through GraphTemporalTransformer for 32-dim structural embeddings.
    - Hybrid feature concatenation (165 raw tabular signals + 32 structural embeddings = 197 dimensions).
    - Cost-sensitive XGBoost probabilistic classification.
    - Contextual SHAP attribution reason code generation for elevated risk scores.
    - 3-tier regulatory risk-triaging evaluation adhering to <100ms payment switch SLA.
    """

    def __init__(
        self,
        graph_path: Optional[str] = None,
        model_path: Optional[str] = None,
        xgb_path: Optional[str] = None,
        config_path: Optional[str] = None,
        low_threshold: float = 0.40,
        high_threshold: float = 0.85,
    ):
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.graph_path = Path(graph_path) if graph_path else (root_dir / "data" / "processed" / "elliptic_pyg_graph.pt")
        self.model_path = (
            Path(model_path) if model_path else (root_dir / "artifacts" / "saved_models" / "graph_transformer_best.pt")
        )
        self.xgb_path = (
            Path(xgb_path) if xgb_path else (root_dir / "artifacts" / "saved_models" / "hybrid_xgboost_best.json")
        )
        self.config_path = (
            Path(config_path) if config_path else (root_dir / "artifacts" / "saved_models" / "model_config.json")
        )

        logger.info("Initializing RealTimeInferencePipeline components...")

        # 1. Load Preprocessed Graph Topology
        if not self.graph_path.exists():
            raise FileNotFoundError(f"PyG graph missing at {self.graph_path}")
        logger.info(f"Loading graph topology from {self.graph_path.name}...")
        self.graph = torch.load(self.graph_path, weights_only=False)

        # 2. Load Graph-Temporal Transformer
        if not self.model_path.exists():
            raise FileNotFoundError(f"Transformer weights missing at {self.model_path}")
        logger.info(f"Loading GraphTemporalTransformer from {self.model_path.name}...")
        self.gtt_model = GraphTemporalTransformer(
            in_channels=self.graph.num_features,
            hidden_channels=64,
            num_heads=4,
            embedding_dim=32,
            out_channels=2,
            dropout=0.0,
        )
        self.gtt_model.load_state_dict(torch.load(self.model_path, weights_only=False))
        self.gtt_model.eval()

        # 3. Load Trained Hybrid XGBoost Model
        if not self.xgb_path.exists():
            raise FileNotFoundError(f"XGBoost model missing at {self.xgb_path}")
        logger.info(f"Loading XGBoost classifier from {self.xgb_path.name}...")
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(self.xgb_path))

        # 4. Initialize SHAP Explainer
        logger.info("Initializing SHAP attribution engine...")
        self.explainer = FraudSHAPExplainer(
            model_path=str(self.xgb_path),
            config_path=str(self.config_path) if self.config_path.exists() else None,
        )

        # 5. Initialize Risk-Triaging Policy Engine
        self.triaging_engine = RiskTriagingEngine(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

        logger.info("RealTimeInferencePipeline ready for live transaction evaluation.")

    def _sample_2hop_subgraph(self, target_node_idx: int):
        """
        Extracts local 2-hop ego-graph around target_node_idx.
        Attempts PyG NeighborLoader, falling back to pure-PyTorch k_hop_subgraph
        capped at [15, 10] neighbors for cross-platform portability.
        """
        try:
            from torch_geometric.loader import NeighborLoader

            loader = NeighborLoader(
                self.graph,
                num_neighbors=[15, 10],
                batch_size=1,
                input_nodes=torch.tensor([target_node_idx]),
                shuffle=False,
            )
            for batch in loader:
                return batch.x, batch.edge_index, 0
        except (ImportError, Exception):
            # Pure PyTorch / PyG k_hop_subgraph fallback (compatible with Python 3.14 without pyg-lib)
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                node_idx=int(target_node_idx),
                num_hops=2,
                edge_index=self.graph.edge_index,
                relabel_nodes=True,
            )
            sub_x = self.graph.x[subset]
            target_pos = mapping.item() if mapping.numel() == 1 else mapping[0].item()
            return sub_x, sub_edge_index, target_pos

    def process_incoming_transaction(self, target_node_idx: int) -> PolicyResponse:
        """
        Executes end-to-end streaming verification for an incoming transaction node:
        1. Samples 2-hop ego-graph around target node.
        2. Computes 32-dim GTT latent structural node embedding.
        3. Combines 165 raw features + 32 structural embeddings -> 197-dim vector.
        4. Predicts fraud probability with XGBoost.
        5. Attributes top-3 SHAP reason codes for elevated risk (prob >= 0.40).
        6. Enforces regulatory 3-tier action policy under <100ms switch SLA.
        """
        t_start = time.perf_counter()

        # Step 1: 2-Hop Ego-Graph Sampling
        sub_x, sub_edge_index, target_pos = self._sample_2hop_subgraph(target_node_idx)

        # Step 2: GTT Structural Embedding Extraction
        with torch.no_grad():
            sub_embeddings, _ = self.gtt_model(sub_x, sub_edge_index)
            target_embedding = sub_embeddings[target_pos : target_pos + 1].cpu().numpy()

        # Step 3: Hybrid Feature Vector Fusion (165 + 32 = 197)
        raw_feat = self.graph.x[target_node_idx : target_node_idx + 1].cpu().numpy()
        hybrid_vector = np.hstack([raw_feat, target_embedding]).astype(np.float32)

        # Step 4: Cost-Sensitive XGBoost Inference
        risk_prob = float(self.xgb_model.predict_proba(hybrid_vector)[0, 1])

        # Step 5: Explainability & SHAP Reason Code Generation
        shap_reasons: List[str] = []
        if risk_prob >= self.triaging_engine.low_threshold:
            explanation = self.explainer.explain_single_transaction(hybrid_vector[0])
            top_pos = explanation.get("top_positive_features", [])[:3]

            for item in top_pos:
                feat = item["feature"]
                delta = item["shap_delta"]
                val = item["feature_value"]
                if item["domain"] == "GNN_Embedding":
                    shap_reasons.append(
                        f"Topological syndicate pattern in {feat} (SHAP impact: +{delta:.3f})"
                    )
                else:
                    shap_reasons.append(
                        f"Abnormal financial velocity on {feat} (Value: {val:.2f}, SHAP impact: +{delta:.3f})"
                    )

            if not shap_reasons:
                shap_reasons = ["Elevated multi-hop risk score breaching institutional tolerance threshold"]
        else:
            shap_reasons = ["Transaction pattern conforms to regular baseline"]

        # Step 6: Latency Measurement & Switch SLA Assessment
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Retrieve original transaction identifier
        tx_id_str = (
            str(self.graph.raw_ids[target_node_idx])
            if hasattr(self.graph, "raw_ids") and target_node_idx < len(self.graph.raw_ids)
            else str(target_node_idx)
        )

        return self.triaging_engine.evaluate(
            tx_id=tx_id_str,
            risk_score=risk_prob,
            shap_reasons=shap_reasons,
            latency_ms=elapsed_ms,
        )
