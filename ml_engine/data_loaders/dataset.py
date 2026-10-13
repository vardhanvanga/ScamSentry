import os
import time
import logging
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Setup module logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class EllipticDatasetLoader:
    """
    Production-grade Data Ingestion and PyTorch Geometric Graph Construction
    for the Elliptic Bitcoin Transaction Network.

    Features:
    - Raw CSV validation and integrity checks.
    - Contiguous 0-indexed integer ID mapping for PyG compatibility.
    - Standardized label sanitization ('1' -> 1 [Illicit], '2' -> 0 [Licit], 'unknown' -> -1 [Background]).
    - Vectorized edge filtering ensuring both source and target exist.
    - Leak-free temporal train/test split masking (Train: timesteps 1-34, Test: timesteps 35-49).
    - High-performance PyG Data serialization and caching.
    """

    DEFAULT_RAW_DIR = "data/raw"
    DEFAULT_PROCESSED_DIR = "data/processed"
    DEFAULT_PROCESSED_FILENAME = "elliptic_pyg_graph.pt"

    CLASSES_CSV = "elliptic_txs_classes.csv"
    EDGELIST_CSV = "elliptic_txs_edgelist.csv"
    FEATURES_CSV = "elliptic_txs_features.csv"

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        processed_filename: str = DEFAULT_PROCESSED_FILENAME,
    ):
        # Resolve repository root
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.raw_dir = Path(raw_dir) if raw_dir else (self.root_dir / self.DEFAULT_RAW_DIR)
        self.processed_dir = Path(processed_dir) if processed_dir else (self.root_dir / self.DEFAULT_PROCESSED_DIR)
        self.processed_file = self.processed_dir / processed_filename

        self._validate_or_recover_raw_files()

    def _validate_or_recover_raw_files(self) -> None:
        """Verify presence of raw Elliptic dataset CSVs, with fallback recovery."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        required_files = [self.CLASSES_CSV, self.EDGELIST_CSV, self.FEATURES_CSV]
        missing_files = [f for f in required_files if not (self.raw_dir / f).exists()]

        if missing_files:
            # Check common fallback locations on the local machine
            user_home = Path.home()
            candidate_dirs = [
                user_home / "Downloads" / "archive" / "elliptic_bitcoin_dataset",
                user_home / "Downloads" / "elliptic_bitcoin_dataset",
                user_home / "OneDrive" / "Desktop" / "archive" / "elliptic_bitcoin_dataset",
            ]

            recovered = False
            for c_dir in candidate_dirs:
                if c_dir.exists() and all((c_dir / f).exists() for f in required_files):
                    logger.info(f"Recovering raw files from candidate archive: {c_dir}")
                    import shutil

                    for f in required_files:
                        src = c_dir / f
                        dst = self.raw_dir / f
                        if not dst.exists():
                            try:
                                os.link(str(src), str(dst))
                                logger.info(f"Hardlinked {f} -> {dst}")
                            except Exception:
                                shutil.copy2(str(src), str(dst))
                                logger.info(f"Copied {f} -> {dst}")
                    recovered = True
                    break

            if not recovered:
                # Re-check if any files are still missing
                still_missing = [f for f in required_files if not (self.raw_dir / f).exists()]
                if still_missing:
                    raise FileNotFoundError(
                        f"Missing required raw Elliptic CSVs in {self.raw_dir}: {still_missing}. "
                        f"Please place {', '.join(required_files)} inside '{self.raw_dir}'."
                    )

        logger.info(f"Raw CSV verification confirmed at: {self.raw_dir}")

    def load_and_process(
        self,
        save_processed: bool = True,
        force_reload: bool = False,
    ) -> Data:
        """
        Constructs or loads cached PyTorch Geometric Data object for the Elliptic Bitcoin dataset.

        Args:
            save_processed: If True, serialize the constructed PyG Data to disk.
            force_reload: If True, bypass cache and recompute from raw CSVs.

        Returns:
            torch_geometric.data.Data object with:
                - x: [num_nodes, 165] (float32)
                - edge_index: [2, num_edges] (int64)
                - y: [num_nodes] (int64)
                - time_step: [num_nodes] (int64)
                - raw_ids: np.ndarray of original string IDs
                - train_mask: [num_nodes] (bool)
                - test_mask: [num_nodes] (bool)
        """
        if not force_reload and self.processed_file.exists():
            logger.info(f"Loading cached PyG graph from: {self.processed_file}")
            t_cache = time.time()
            data = torch.load(self.processed_file, weights_only=False)
            load_sec = time.time() - t_cache
            logger.info(
                f"Successfully loaded cached graph in {load_sec:.2f}s "
                f"({data.num_nodes:,} nodes, {data.num_edges:,} edges)"
            )
            return data

        logger.info("Initiating full ingestion and dynamic graph construction pipeline...")
        t_start = time.time()

        # -------------------------------------------------------------
        # 1. Ingest Classes
        # -------------------------------------------------------------
        t0 = time.time()
        classes_path = self.raw_dir / self.CLASSES_CSV
        logger.info(f"Reading transaction classes from {classes_path.name}...")
        df_classes = pd.read_csv(classes_path)

        # Clean labels: '1' -> 1 (Illicit), '2' -> 0 (Licit), 'unknown' -> -1 (Unlabeled)
        label_mapping = {
            "1": 1,
            "2": 0,
            "unknown": -1,
            1: 1,
            2: 0,
            -1: -1,
        }
        df_classes["clean_label"] = df_classes["class"].map(label_mapping).fillna(-1).astype(np.int64)
        logger.info(
            f"Parsed {len(df_classes):,} classes in {time.time() - t0:.2f}s "
            f"(Illicit: {(df_classes['clean_label'] == 1).sum():,}, "
            f"Licit: {(df_classes['clean_label'] == 0).sum():,}, "
            f"Unknown: {(df_classes['clean_label'] == -1).sum():,})"
        )

        # -------------------------------------------------------------
        # 2. Ingest Node Features & Timesteps
        # -------------------------------------------------------------
        t0 = time.time()
        features_path = self.raw_dir / self.FEATURES_CSV
        logger.info(f"Reading node features from {features_path.name} (165 features + timestep + txId)...")

        # Explicit dtypes to minimize memory footprint and accelerate CSV parsing
        dtypes: Dict[int, Any] = {0: np.int64, 1: np.int64}
        for col_idx in range(2, 167):
            dtypes[col_idx] = np.float32

        df_features = pd.read_csv(features_path, header=None, dtype=dtypes)
        num_nodes = len(df_features)
        num_features = df_features.shape[1] - 2  # subtract txId and timestep
        logger.info(f"Parsed {num_nodes:,} nodes with {num_features} features in {time.time() - t0:.2f}s")

        # -------------------------------------------------------------
        # 3. Contiguous 0-indexed ID Mapping
        # -------------------------------------------------------------
        t0 = time.time()
        raw_tx_ids = df_features[0].values
        raw_ids_str = raw_tx_ids.astype(str)
        id_to_idx = {tx_id: idx for idx, tx_id in enumerate(raw_tx_ids)}
        logger.info(f"Constructed contiguous 0-indexed ID mapping [0..{num_nodes - 1}] in {time.time() - t0:.2f}s")

        # Extract features and timestep tensors
        x_np = df_features.iloc[:, 2:167].values.astype(np.float32)
        timestep_np = df_features[1].values.astype(np.int64)

        # Align labels to the feature-row ordering
        class_series = df_classes.set_index("txId")["clean_label"]
        y_np = class_series.reindex(df_features[0]).fillna(-1).values.astype(np.int64)

        # -------------------------------------------------------------
        # 4. Ingest and Filter Edgelist
        # -------------------------------------------------------------
        t0 = time.time()
        edgelist_path = self.raw_dir / self.EDGELIST_CSV
        logger.info(f"Reading edges from {edgelist_path.name}...")
        df_edges = pd.read_csv(edgelist_path, dtype={"txId1": np.int64, "txId2": np.int64})
        initial_edges = len(df_edges)

        # Filter edges where both source and target exist in the feature node set
        valid_mask = df_edges["txId1"].isin(id_to_idx) & df_edges["txId2"].isin(id_to_idx)
        df_filtered_edges = df_edges[valid_mask]
        num_valid_edges = len(df_filtered_edges)

        if num_valid_edges < initial_edges:
            logger.warning(f"Filtered out {initial_edges - num_valid_edges:,} dangling edges outside node set.")

        src_indices = df_filtered_edges["txId1"].map(id_to_idx).values.astype(np.int64)
        dst_indices = df_filtered_edges["txId2"].map(id_to_idx).values.astype(np.int64)

        edge_index_np = np.stack([src_indices, dst_indices], axis=0)
        logger.info(f"Processed {num_valid_edges:,} directed edges in {time.time() - t0:.2f}s")

        # -------------------------------------------------------------
        # 5. Build PyTorch Tensors & Leak-Free Temporal Masks
        # -------------------------------------------------------------
        t0 = time.time()
        x_tensor = torch.from_numpy(x_np)
        edge_index_tensor = torch.from_numpy(edge_index_np).long()
        y_tensor = torch.from_numpy(y_np).long()
        time_step_tensor = torch.from_numpy(timestep_np).long()

        # Leak-Free Temporal Train/Test Split
        # Train: timesteps 1-34, only labeled nodes (y != -1)
        # Test:  timesteps 35-49, only labeled nodes (y != -1)
        # Background nodes (y == -1) remain in graph structure for message passing
        train_mask = (time_step_tensor >= 1) & (time_step_tensor <= 34) & (y_tensor != -1)
        test_mask = (time_step_tensor >= 35) & (time_step_tensor <= 49) & (y_tensor != -1)

        # -------------------------------------------------------------
        # 6. Instantiate PyG Data Container
        # -------------------------------------------------------------
        data = Data(
            x=x_tensor,
            edge_index=edge_index_tensor,
            y=y_tensor,
            time_step=time_step_tensor,
            train_mask=train_mask,
            test_mask=test_mask,
        )
        data.raw_ids = raw_ids_str

        # -------------------------------------------------------------
        # 7. Summary Metrics & Class Balance Logging
        # -------------------------------------------------------------
        train_y = y_tensor[train_mask]
        test_y = y_tensor[test_mask]

        train_illicit = (train_y == 1).sum().item()
        train_licit = (train_y == 0).sum().item()
        test_illicit = (test_y == 1).sum().item()
        test_licit = (test_y == 0).sum().item()

        logger.info("=" * 60)
        logger.info("ScamSentry Elliptic Graph Ingestion Summary")
        logger.info("=" * 60)
        logger.info(f"Total Nodes:            {data.num_nodes:,}")
        logger.info(f"Total Edges:            {data.num_edges:,}")
        logger.info(f"Feature Dimensions:     {data.num_features}")
        logger.info(f"Total Timesteps:        {int(time_step_tensor.min())} to {int(time_step_tensor.max())}")
        logger.info(
            f"Train Set (Steps 1-34): {train_mask.sum().item():,} nodes "
            f"| Illicit: {train_illicit:,} ({(train_illicit/len(train_y)*100):.2f}%), "
            f"Licit: {train_licit:,} ({(train_licit/len(train_y)*100):.2f}%)"
        )
        logger.info(
            f"Test Set (Steps 35-49): {test_mask.sum().item():,} nodes "
            f"| Illicit: {test_illicit:,} ({(test_illicit/len(test_y)*100):.2f}%), "
            f"Licit: {test_licit:,} ({(test_licit/len(test_y)*100):.2f}%)"
        )
        logger.info(f"Background (Unlabeled): {(y_tensor == -1).sum().item():,} nodes (retained for GNN aggregation)")
        logger.info("=" * 60)

        # -------------------------------------------------------------
        # 8. Caching
        # -------------------------------------------------------------
        if save_processed:
            logger.info(f"Serializing processed PyG graph to {self.processed_file}...")
            torch.save(data, self.processed_file)
            size_mb = self.processed_file.stat().st_size / (1024 * 1024)
            logger.info(f"Serialized graph successfully ({size_mb:.2f} MB)")

        total_sec = time.time() - t_start
        logger.info(f"Complete pipeline executed in {total_sec:.2f} seconds.")

        return data


if __name__ == "__main__":
    loader = EllipticDatasetLoader()
    graph = loader.load_and_process(save_processed=True)
    print("Graph construction complete:", graph)
