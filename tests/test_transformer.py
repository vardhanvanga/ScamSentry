import unittest
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_engine.models.graph_transformer import GraphTemporalTransformer
from ml_engine.models.focal_loss import FocalLoss


class TestGraphTemporalTransformer(unittest.TestCase):
    """Automated assertions for Phase 2 Graph-Temporal Transformer & Focal Loss."""

    def setUp(self):
        self.num_nodes = 20
        self.in_channels = 165
        self.embedding_dim = 32
        self.num_classes = 2

        # Create dummy synthetic graph tensors
        self.x = torch.randn(self.num_nodes, self.in_channels)
        # Create random directed edges
        src = torch.randint(0, self.num_nodes, (40,))
        dst = torch.randint(0, self.num_nodes, (40,))
        self.edge_index = torch.stack([src, dst], dim=0)

        # Labels: 0 (licit), 1 (illicit), -1 (unlabeled)
        self.y = torch.tensor([0, 1, 0, 0, 1, -1, 0, 1, -1, 0, 0, 1, 0, 0, -1, 1, 0, 0, 1, 0])
        self.train_mask = torch.tensor([True] * 10 + [False] * 10)
        self.test_mask = torch.tensor([False] * 10 + [True] * 10)

    def test_01_model_initialization(self):
        """1. Assert model initializes with correct layer dimensions."""
        model = GraphTemporalTransformer(
            in_channels=self.in_channels,
            hidden_channels=64,
            num_heads=4,
            embedding_dim=self.embedding_dim,
            out_channels=self.num_classes,
        )
        self.assertEqual(model.in_channels, 165)
        self.assertEqual(model.hidden_channels, 64)
        self.assertEqual(model.num_heads, 4)
        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.out_channels, 2)

    def test_02_forward_pass_shapes(self):
        """2. Assert single forward pass produces [N, 32] embeddings and [N, 2] logits."""
        model = GraphTemporalTransformer(
            in_channels=self.in_channels,
            hidden_channels=64,
            num_heads=4,
            embedding_dim=self.embedding_dim,
            out_channels=self.num_classes,
        )
        model.eval()
        with torch.no_grad():
            embeddings, logits = model(self.x, self.edge_index)

        self.assertEqual(
            embeddings.shape,
            torch.Size([self.num_nodes, self.embedding_dim]),
            f"Expected embeddings shape [{self.num_nodes}, 32], got {embeddings.shape}",
        )
        self.assertEqual(
            logits.shape,
            torch.Size([self.num_nodes, self.num_classes]),
            f"Expected logits shape [{self.num_nodes}, 2], got {logits.shape}",
        )

    def test_03_focal_loss_masking_and_gradient(self):
        """3. Assert FocalLoss respects train_mask, ignores -1 targets, and computes gradients."""
        criterion = FocalLoss(alpha=0.85, gamma=2.0, reduction="mean")
        logits = torch.randn(self.num_nodes, 2, requires_grad=True)

        loss = criterion(logits, self.y, mask=self.train_mask)
        self.assertFalse(torch.isnan(loss), "FocalLoss returned NaN")
        self.assertGreater(loss.item(), 0.0, "FocalLoss should be positive")

        loss.backward()
        self.assertIsNotNone(logits.grad, "Gradients should be computed for logits")
        # Check that nodes outside train_mask or with target == -1 have zero gradients
        for i in range(self.num_nodes):
            if not self.train_mask[i] or self.y[i] == -1:
                self.assertTrue(
                    torch.allclose(logits.grad[i], torch.zeros_like(logits.grad[i])),
                    f"Node {i} outside train_mask or with target -1 received non-zero gradient!",
                )

    def test_04_training_loop_loss_decrease(self):
        """4. Assert loss strictly decreases over a 3-epoch optimization run."""
        model = GraphTemporalTransformer(
            in_channels=self.in_channels,
            hidden_channels=64,
            num_heads=4,
            embedding_dim=self.embedding_dim,
            out_channels=self.num_classes,
            dropout=0.0,  # Deterministic for test
        )
        criterion = FocalLoss(alpha=0.85, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        losses = []
        for _ in range(3):
            model.train()
            optimizer.zero_grad()
            _, logits = model(self.x, self.edge_index)
            loss = criterion(logits, self.y, mask=self.train_mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(
            losses[-1],
            losses[0],
            f"Expected loss to decrease over 3 epochs, but went from {losses[0]:.4f} to {losses[-1]:.4f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
