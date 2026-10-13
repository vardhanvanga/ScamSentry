import unittest
import time
import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml_engine.data_loaders.dataset import EllipticDatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestEllipticDataset")


class TestEllipticDataset(unittest.TestCase):
    """Automated assertions and integrity verification for Phase 1 Elliptic Graph Pipeline."""

    @classmethod
    def setUpClass(cls):
        """Construct or load graph once for the entire test suite."""
        logger.info("Initializing EllipticDatasetLoader for test suite...")
        t0 = time.time()
        cls.loader = EllipticDatasetLoader()
        cls.graph = cls.loader.load_and_process(save_processed=True, force_reload=False)
        cls.load_duration = time.time() - t0
        logger.info(f"Graph loaded into memory in {cls.load_duration:.2f} seconds.")

    def test_01_graph_dimensions(self):
        """1. Assert graph node count and feature dimensions."""
        logger.info("Asserting graph node count and feature dimensionality...")
        self.assertEqual(
            self.graph.num_nodes,
            203769,
            f"Expected 203,769 nodes, but got {self.graph.num_nodes}",
        )
        self.assertEqual(
            self.graph.x.shape[1],
            165,
            f"Expected 165 node features, but got {self.graph.x.shape[1]}",
        )
        self.assertEqual(
            self.graph.x.shape[0],
            203769,
            f"Expected feature matrix rows to match node count 203,769, got {self.graph.x.shape[0]}",
        )
        self.assertEqual(
            len(self.graph.raw_ids),
            203769,
            f"Expected 203,769 raw IDs, got {len(self.graph.raw_ids)}",
        )
        logger.info("Assertion PASSED: Exactly 203,769 nodes with 165 features validated.")

    def test_02_edge_index_bounds(self):
        """2. Assert edge indices are non-negative and strictly within [0, num_nodes - 1]."""
        logger.info("Asserting edge tensor out-of-bounds safety...")
        max_edge_idx = self.graph.edge_index.max().item()
        min_edge_idx = self.graph.edge_index.min().item()

        self.assertGreaterEqual(
            min_edge_idx,
            0,
            f"Edge index contains negative indices: {min_edge_idx}",
        )
        self.assertLess(
            max_edge_idx,
            self.graph.num_nodes,
            f"Edge index out of bounds: max index {max_edge_idx} >= num_nodes {self.graph.num_nodes}",
        )
        self.assertEqual(
            self.graph.edge_index.shape[0],
            2,
            f"Edge index must be of shape [2, E], got shape {self.graph.edge_index.shape}",
        )
        logger.info(
            f"Assertion PASSED: All {self.graph.num_edges:,} edges safely index within [0, {self.graph.num_nodes - 1}]."
        )

    def test_03_leak_free_temporal_split(self):
        """3. Assert leak-free temporal separation between train and test partitions."""
        logger.info("Asserting temporal leak-free separation...")
        train_steps = self.graph.time_step[self.graph.train_mask]
        test_steps = self.graph.time_step[self.graph.test_mask]

        max_train_step = train_steps.max().item()
        min_test_step = test_steps.min().item()

        # Strict temporal split assertion
        self.assertLessEqual(
            max_train_step,
            34,
            f"Temporal leakage detected: max train timestep {max_train_step} > 34",
        )
        self.assertGreaterEqual(
            min_test_step,
            35,
            f"Temporal leakage detected: min test timestep {min_test_step} < 35",
        )

        # Mutually exclusive masks
        overlap = (self.graph.train_mask & self.graph.test_mask).sum().item()
        self.assertEqual(
            overlap,
            0,
            f"Train mask and test mask must be strictly disjoint, but found {overlap} overlapping nodes.",
        )

        # Ensure no unlabeled background nodes exist in evaluation masks
        train_unlabeled = (self.graph.y[self.graph.train_mask] == -1).sum().item()
        test_unlabeled = (self.graph.y[self.graph.test_mask] == -1).sum().item()
        self.assertEqual(
            train_unlabeled,
            0,
            f"Found {train_unlabeled} unlabeled nodes in train_mask.",
        )
        self.assertEqual(
            test_unlabeled,
            0,
            f"Found {test_unlabeled} unlabeled nodes in test_mask.",
        )

        logger.info(
            f"Assertion PASSED: Train timesteps max={max_train_step} (<= 34), "
            f"Test timesteps min={min_test_step} (>= 35). Zero temporal leakage."
        )

    def test_04_class_distribution_and_imbalance(self):
        """4. Assert both classes exist in both splits and report class imbalance metrics."""
        logger.info("Asserting class presence and computing imbalance ratios...")
        train_labels = self.graph.y[self.graph.train_mask]
        test_labels = self.graph.y[self.graph.test_mask]

        train_illicit = (train_labels == 1).sum().item()
        train_licit = (train_labels == 0).sum().item()

        test_illicit = (test_labels == 1).sum().item()
        test_licit = (test_labels == 0).sum().item()

        self.assertGreater(train_illicit, 0, "Train split contains 0 Illicit (class 1) transactions.")
        self.assertGreater(train_licit, 0, "Train split contains 0 Licit (class 0) transactions.")
        self.assertGreater(test_illicit, 0, "Test split contains 0 Illicit (class 1) transactions.")
        self.assertGreater(test_licit, 0, "Test split contains 0 Licit (class 0) transactions.")

        train_imbalance = train_licit / train_illicit if train_illicit > 0 else float("inf")
        test_imbalance = test_licit / test_illicit if test_illicit > 0 else float("inf")

        logger.info("-" * 55)
        logger.info("TEMPORAL DATASET CLASS AUDIT METRICS:")
        logger.info(f"  • Train Total Labeled: {len(train_labels):,}")
        logger.info(f"    - Illicit (Fraud = 1): {train_illicit:,} ({train_illicit/len(train_labels)*100:.2f}%)")
        logger.info(f"    - Licit   (Normal = 0): {train_licit:,} ({train_licit/len(train_labels)*100:.2f}%)")
        logger.info(f"    - Class Imbalance Ratio: {train_imbalance:.2f} : 1")
        logger.info(f"  • Test Total Labeled:  {len(test_labels):,}")
        logger.info(f"    - Illicit (Fraud = 1): {test_illicit:,} ({test_illicit/len(test_labels)*100:.2f}%)")
        logger.info(f"    - Licit   (Normal = 0): {test_licit:,} ({test_licit/len(test_labels)*100:.2f}%)")
        logger.info(f"    - Class Imbalance Ratio: {test_imbalance:.2f} : 1")
        logger.info("-" * 55)

        logger.info("Assertion PASSED: Both classes robustly populated across all temporal splits.")

    def test_05_caching_performance(self):
        """5. Assert cached PyG graph loads rapidly from disk."""
        logger.info("Asserting cache reload latency...")
        cache_path = self.loader.processed_file
        self.assertTrue(cache_path.exists(), f"Processed graph cache missing at: {cache_path}")

        t0 = time.time()
        reloaded_graph = self.loader.load_and_process(save_processed=False, force_reload=False)
        reload_time = time.time() - t0

        self.assertEqual(reloaded_graph.num_nodes, self.graph.num_nodes)
        self.assertEqual(reloaded_graph.num_edges, self.graph.num_edges)
        self.assertLess(
            reload_time,
            5.0,
            f"Cache reload took {reload_time:.2f}s, expected < 5.0s",
        )
        logger.info(f"Assertion PASSED: Cached PyG graph reloaded in {reload_time:.3f}s (< 5.0s limit).")


def run_verification():
    """CLI runner executing the test suite with timing statistics."""
    start_time = time.time()
    logger.info("================================================================")
    logger.info("   SCAMSENTRY (STAR-FRAUD ENGINE) — PHASE 1 PIPELINE VERIFICATION")
    logger.info("================================================================")

    suite = unittest.TestLoader().loadTestsFromTestCase(TestEllipticDataset)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total_time = time.time() - start_time
    logger.info("================================================================")
    logger.info(f"Execution finished in {total_time:.2f} seconds.")
    if result.wasSuccessful():
        logger.info("STATUS: ALL PHASE 1 PIPELINE ASSERTIONS PASSED SUCCESSFULLY!")
    else:
        logger.error(f"STATUS: TEST SUITE FAILED with {len(result.failures)} failures and {len(result.errors)} errors.")
    logger.info("================================================================")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_verification())
