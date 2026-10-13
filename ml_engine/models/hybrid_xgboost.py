import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_and_fuse_features(
    graph_path: Optional[str] = None,
    embeddings_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], float, np.ndarray]:
    """
    Loads raw node features and GTT-derived structural node embeddings, performs
    horizontal concatenation into a 197-dimensional hybrid feature matrix, and
    partitions into temporal train/test sets.

    Args:
        graph_path: Path to cached PyG graph object (.pt).
        embeddings_path: Path to cached Graph-Temporal Transformer embeddings tensor (.pt).

    Returns:
        Tuple of:
            - X_train: Training features [N_train, 197]
            - y_train: Training labels [N_train]
            - X_test: Testing features [N_test, 197]
            - y_test: Testing labels [N_test]
            - feature_names: List of 197 column names
            - scale_pos_weight: Imbalance ratio (licit / illicit) in train set
            - test_tx_ids: Array of raw transaction IDs corresponding to test records
    """
    root_dir = Path(__file__).resolve().parent.parent.parent
    graph_file = Path(graph_path) if graph_path else (root_dir / "data" / "processed" / "elliptic_pyg_graph.pt")
    embeddings_file = (
        Path(embeddings_path) if embeddings_path else (root_dir / "artifacts" / "saved_models" / "graph_embeddings.pt")
    )

    if not graph_file.exists():
        raise FileNotFoundError(f"PyG graph file not found at: {graph_file}")
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Graph embeddings file not found at: {embeddings_file}")

    logger.info(f"Loading PyG graph from: {graph_file}")
    graph = torch.load(graph_file, weights_only=False)

    logger.info(f"Loading structural node embeddings from: {embeddings_file}")
    embeddings = torch.load(embeddings_file, weights_only=True)

    # Integrity assertions
    assert (
        graph.x.shape[0] == embeddings.shape[0] == 203769
    ), f"Mismatch in node count: graph.x has {graph.x.shape[0]}, embeddings has {embeddings.shape[0]}"
    assert (
        graph.x.shape[1] == 165
    ), f"Expected 165 raw features, got {graph.x.shape[1]}"
    assert (
        embeddings.shape[1] == 32
    ), f"Expected 32 structural embeddings, got {embeddings.shape[1]}"

    # Horizontal concatenation: 165 raw features + 32 GNN embeddings = 197 dimensions
    x_raw_np = graph.x.cpu().numpy().astype(np.float32)
    embs_np = embeddings.cpu().numpy().astype(np.float32)
    X_all = np.hstack([x_raw_np, embs_np])
    logger.info(f"Fused hybrid feature matrix constructed: shape {X_all.shape}")

    # Column nomenclature
    raw_feature_names = [f"raw_feat_{i}" for i in range(165)]
    graph_emb_names = [f"graph_emb_{i}" for i in range(32)]
    feature_names = raw_feature_names + graph_emb_names
    assert len(feature_names) == 197

    # Extract temporal splits strictly using the masks
    train_indices = graph.train_mask.cpu().numpy()
    test_indices = graph.test_mask.cpu().numpy()

    X_train = X_all[train_indices]
    y_train = graph.y.cpu().numpy()[train_indices].astype(np.int64)

    X_test = X_all[test_indices]
    y_test = graph.y.cpu().numpy()[test_indices].astype(np.int64)

    raw_ids_np = np.array(graph.raw_ids)
    test_tx_ids = raw_ids_np[test_indices]

    # Verify no unlabeled background nodes exist in evaluation splits
    assert not np.any(y_train == -1), "Train split contains unlabeled background nodes (-1)!"
    assert not np.any(y_test == -1), "Test split contains unlabeled background nodes (-1)!"

    # Check for NaNs or Infs
    assert not np.isnan(X_train).any(), "X_train contains NaNs!"
    assert not np.isnan(X_test).any(), "X_test contains NaNs!"
    assert not np.isinf(X_train).any(), "X_train contains Infs!"
    assert not np.isinf(X_test).any(), "X_test contains Infs!"

    # Compute dynamic cost-sensitive imbalance weight
    num_licit = np.sum(y_train == 0)
    num_illicit = np.sum(y_train == 1)
    scale_pos_weight = float(num_licit) / float(max(1, num_illicit))

    logger.info("=" * 60)
    logger.info("Hybrid Tabular-Structural Feature Fusion Summary")
    logger.info("=" * 60)
    logger.info(f"Total Transactions:      {X_all.shape[0]:,}")
    logger.info(f"Combined Dimensions:     {X_all.shape[1]} (165 Tabular + 32 Structural)")
    logger.info(f"Train Set (Steps 1-34):  {X_train.shape[0]:,} samples")
    logger.info(f"  - Illicit (1):         {num_illicit:,} ({(num_illicit / len(y_train) * 100):.2f}%)")
    logger.info(f"  - Licit (0):           {num_licit:,} ({(num_licit / len(y_train) * 100):.2f}%)")
    logger.info(f"  - Dynamic Imbalance Wt:{scale_pos_weight:.4f}")
    logger.info(f"Test Set (Steps 35-49):  {X_test.shape[0]:,} samples")
    logger.info(f"  - Illicit (1):         {np.sum(y_test == 1):,} ({(np.sum(y_test == 1) / len(y_test) * 100):.2f}%)")
    logger.info(f"  - Licit (0):           {np.sum(y_test == 0):,} ({(np.sum(y_test == 0) / len(y_test) * 100):.2f}%)")
    logger.info("=" * 60)

    return X_train, y_train, X_test, y_test, feature_names, scale_pos_weight, test_tx_ids


class HybridXGBoostClassifier:
    """
    Production wrapper for the Hybrid XGBoost Classifier combining raw tabular
    transaction signals with Graph-Temporal Transformer structural node embeddings.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 2,
        scale_pos_weight: float = 7.63,
        gamma: float = 0.1,
        early_stopping_rounds: int = 25,
        random_state: int = 42,
        n_jobs: int = -1,
        tree_method: str = "hist",
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "scale_pos_weight": scale_pos_weight,
            "gamma": gamma,
            "eval_metric": ["logloss", "aucpr"],
            "early_stopping_rounds": early_stopping_rounds,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "tree_method": tree_method,
        }
        self.model = xgb.XGBClassifier(**self.params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: Union[bool, int] = 50,
    ):
        """Fit model with early stopping on evaluation set."""
        return self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classes with custom decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def save_model(self, file_path: str):
        """Persist model in JSON format."""
        self.model.save_model(file_path)

    def load_model(self, file_path: str):
        """Load model from JSON format."""
        self.model.load_model(file_path)
