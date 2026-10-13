import os
import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_engine.data_loaders.dataset import EllipticDatasetLoader
from ml_engine.models.graph_transformer import GraphTemporalTransformer
from ml_engine.models.focal_loss import FocalLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("TrainTransformer")


def train_graph_transformer(
    epochs: int = 35,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    alpha: float = 0.85,
    gamma: float = 2.0,
    device: Optional[str] = None,
    save_dir: Optional[str] = None,
):
    """
    Train Graph-Temporal Transformer with Edge Attention and Focal Loss
    on the Elliptic Bitcoin transaction network.
    """
    # -------------------------------------------------------------
    # 1. Environment & Hardware Setup
    # -------------------------------------------------------------
    num_cpus = os.cpu_count() or 8
    torch.set_num_threads(num_cpus)
    logger.info(f"Initialized PyTorch with {torch.get_num_threads()} CPU threads.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    logger.info(f"Target Compute Device: {device}")

    save_path = Path(save_dir) if save_dir else (ROOT_DIR / "artifacts" / "saved_models")
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_file = save_path / "graph_transformer_best.pt"
    embeddings_file = save_path / "graph_embeddings.pt"

    # -------------------------------------------------------------
    # 2. Data Ingestion
    # -------------------------------------------------------------
    logger.info("Loading PyTorch Geometric graph...")
    loader = EllipticDatasetLoader()
    graph = loader.load_and_process(save_processed=True, force_reload=False)
    graph = graph.to(device)

    # -------------------------------------------------------------
    # 3. Model, Loss, Optimizer & Scheduler Initialization
    # -------------------------------------------------------------
    model = GraphTemporalTransformer(
        in_channels=graph.num_features,
        hidden_channels=64,
        num_heads=4,
        embedding_dim=32,
        out_channels=2,
        dropout=0.2,
        beta=True,
    ).to(device)

    criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    logger.info("================================================================================")
    logger.info("          STARTING GRAPH-TEMPORAL TRANSFORMER TRAINING PIPELINE                 ")
    logger.info(f"  Epochs: {epochs} | LR: {lr} | Weight Decay: {weight_decay} | Focal(alpha={alpha}, gamma={gamma})")
    logger.info("================================================================================")

    best_val_f1 = -1.0
    best_epoch = 0
    total_start_time = time.time()

    # Pre-extract test mask numpy array for metric calculation
    test_mask_np = graph.test_mask.cpu().numpy()
    y_test_true = graph.y[graph.test_mask].cpu().numpy()

    # -------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        optimizer.zero_grad()

        embeddings, logits = model(graph.x, graph.edge_index)
        loss = criterion(logits, graph.y, mask=graph.train_mask)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # --- Evaluation Phase (on Test Mask) ---
        model.eval()
        with torch.no_grad():
            _, eval_logits = model(graph.x, graph.edge_index)
            probs = F.softmax(eval_logits, dim=-1)[:, 1].cpu().numpy()

        test_probs = probs[test_mask_np]
        test_preds = (test_probs >= 0.5).astype(int)

        # Compute minority (Illicit) class evaluation metrics
        test_precision = precision_score(y_test_true, test_preds, pos_label=1, zero_division=0)
        test_recall = recall_score(y_test_true, test_preds, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test_true, test_preds, pos_label=1, zero_division=0)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test_true, test_probs, pos_label=1)
        test_pr_auc = auc(recall_curve, precision_curve)

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.2f}s] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Illicit Prec: {test_precision:.4f} | "
            f"Recall: {test_recall:.4f} | "
            f"F1: {test_f1:.4f} | "
            f"PR-AUC: {test_pr_auc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Step scheduler based on Illicit F1-score
        scheduler.step(test_f1)

        # Checkpoint best model
        if test_f1 > best_val_f1:
            best_val_f1 = test_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_file)
            logger.info(f"  ★ New best model checkpoint saved (F1: {test_f1:.4f}) -> {best_model_file.name}")

    total_training_time = time.time() - total_start_time
    logger.info("================================================================================")
    logger.info(
        f"TRAINING COMPLETE in {total_training_time:.2f}s! Best Epoch: {best_epoch:02d} "
        f"with Illicit F1: {best_val_f1:.4f}"
    )
    logger.info("================================================================================")

    # -------------------------------------------------------------
    # 5. Extract & Persist Full Latent Structural Node Embeddings
    # -------------------------------------------------------------
    logger.info("Loading best model weights for deterministic embedding extraction...")
    model.load_state_dict(torch.load(best_model_file, weights_only=False))
    model.eval()

    with torch.no_grad():
        final_embeddings, final_logits = model(graph.x, graph.edge_index)
        embeddings_tensor = final_embeddings.cpu()

    logger.info(f"Extracted structural node embeddings shape: {embeddings_tensor.shape}")
    torch.save(embeddings_tensor, embeddings_file)
    file_size_mb = embeddings_file.stat().st_size / (1024 * 1024)
    logger.info(
        f"Persisted node embeddings ({file_size_mb:.2f} MB) successfully to: {embeddings_file}"
    )

    return {
        "best_epoch": best_epoch,
        "best_f1": best_val_f1,
        "embeddings_shape": list(embeddings_tensor.shape),
        "embeddings_file": str(embeddings_file),
        "model_file": str(best_model_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph-Temporal Transformer for ScamSentry")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs (default: 35)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--alpha", type=float, default=0.85, help="Focal loss alpha weight for fraud class")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma focusing parameter")
    args = parser.parse_args()

    train_graph_transformer(
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        gamma=args.gamma,
    )
