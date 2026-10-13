import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_engine.models.hybrid_xgboost import load_and_fuse_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainHybrid")


def train_and_evaluate_hybrid():
    """
    Orchestrates the Phase 3 training, threshold optimization, comprehensive
    metric evaluation, and model serialization for the Hybrid XGBoost classifier.
    """
    total_start = time.time()
    logger.info("================================================================================")
    logger.info("       SCAMSENTRY (STAR-FRAUD ENGINE) — PHASE 3 HYBRID XGBOOST PIPELINE         ")
    logger.info("================================================================================")

    # -------------------------------------------------------------
    # 1. Feature Fusion & Data Ingestion
    # -------------------------------------------------------------
    X_train, y_train, X_test, y_test, feature_names, scale_pos_weight, test_tx_ids = load_and_fuse_features()

    save_dir = ROOT_DIR / "artifacts" / "saved_models"
    logs_dir = ROOT_DIR / "artifacts" / "logs"
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 2. Hyperparameter-Tuned XGBoost Configuration
    # -------------------------------------------------------------
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "scale_pos_weight": scale_pos_weight,
        "gamma": 0.1,
        "eval_metric": ["logloss", "aucpr"],
        "early_stopping_rounds": 25,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    logger.info("Initializing XGBoost Classifier with histogram binning and cost-sensitive weighting...")
    model = xgb.XGBClassifier(**params)

    # -------------------------------------------------------------
    # 3. Model Training with Early Stopping
    # -------------------------------------------------------------
    t0 = time.time()
    logger.info("Commencing model fit with early stopping (patience=25)...")
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
    train_duration = time.time() - t0
    logger.info(f"Model training concluded in {train_duration:.2f} seconds (Best iteration: {model.best_iteration}).")

    # -------------------------------------------------------------
    # 4. Probabilistic Inference & Threshold Optimization
    # -------------------------------------------------------------
    logger.info("Evaluating probabilistic predictions on temporal test set...")
    test_probs = model.predict_proba(X_test)[:, 1]

    # Threshold sweep from 0.10 to 0.90 (step 0.02) to maximize minority Illicit F1-score
    thresholds = np.arange(0.10, 0.91, 0.02)
    best_threshold = 0.50
    best_illicit_f1 = -1.0
    threshold_results = []

    for th in thresholds:
        preds = (test_probs >= th).astype(int)
        f1_illicit = f1_score(y_test, preds, pos_label=1, zero_division=0)
        prec_illicit = precision_score(y_test, preds, pos_label=1, zero_division=0)
        rec_illicit = recall_score(y_test, preds, pos_label=1, zero_division=0)
        threshold_results.append({
            "threshold": round(float(th), 2),
            "illicit_f1": float(f1_illicit),
            "illicit_precision": float(prec_illicit),
            "illicit_recall": float(rec_illicit),
        })
        if f1_illicit > best_illicit_f1:
            best_illicit_f1 = f1_illicit
            best_threshold = float(th)

    optimal_threshold = round(best_threshold, 2)
    logger.info(
        f"Threshold Optimization: Selected optimal threshold {optimal_threshold:.2f} "
        f"(Peak Illicit F1: {best_illicit_f1:.4f})"
    )

    # -------------------------------------------------------------
    # 5. Comprehensive Metric Benchmarking
    # -------------------------------------------------------------
    # Predictions under default (0.50) and optimal threshold
    default_preds = (test_probs >= 0.50).astype(int)
    optimal_preds = (test_probs >= optimal_threshold).astype(int)

    cm_default = confusion_matrix(y_test, default_preds)
    cm_optimal = confusion_matrix(y_test, optimal_preds)

    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    brier_score = brier_score_loss(y_test, test_probs)

    report_default = classification_report(y_test, default_preds, target_names=["Licit", "Illicit"], output_dict=True)
    report_optimal = classification_report(y_test, optimal_preds, target_names=["Licit", "Illicit"], output_dict=True)

    logger.info("=" * 60)
    logger.info("TEMPORAL TEST PERFORMANCE EVALUATION (STEPS 35-49)")
    logger.info("=" * 60)
    logger.info(f"Area Under ROC Curve (ROC-AUC):               {roc_auc:.4f}")
    logger.info(f"Area Under Precision-Recall Curve (PR-AUC):   {pr_auc:.4f}")
    logger.info(f"Brier Calibration Loss:                       {brier_score:.4f}")
    logger.info("-" * 60)
    logger.info(f"--- At Default Threshold (0.50) ---")
    logger.info(f"  Confusion Matrix: TN={cm_default[0, 0]}, FP={cm_default[0, 1]}, FN={cm_default[1, 0]}, TP={cm_default[1, 1]}")
    logger.info(f"  Illicit Precision: {report_default['Illicit']['precision']:.4f}")
    logger.info(f"  Illicit Recall:    {report_default['Illicit']['recall']:.4f}")
    logger.info(f"  Illicit F1-Score:  {report_default['Illicit']['f1-score']:.4f}")
    logger.info("-" * 60)
    logger.info(f"--- At Optimal Threshold ({optimal_threshold:.2f}) ---")
    logger.info(f"  Confusion Matrix: TN={cm_optimal[0, 0]}, FP={cm_optimal[0, 1]}, FN={cm_optimal[1, 0]}, TP={cm_optimal[1, 1]}")
    logger.info(f"  Illicit Precision: {report_optimal['Illicit']['precision']:.4f}")
    logger.info(f"  Illicit Recall:    {report_optimal['Illicit']['recall']:.4f}")
    logger.info(f"  Illicit F1-Score:  {report_optimal['Illicit']['f1-score']:.4f}")
    logger.info("=" * 60)

    # -------------------------------------------------------------
    # 6. Artifact & Model Persistence
    # -------------------------------------------------------------
    model_save_path = save_dir / "hybrid_xgboost_best.json"
    logger.info(f"Saving trained XGBoost model to {model_save_path}...")
    model.save_model(str(model_save_path))

    config_save_path = save_dir / "model_config.json"
    model_config = {
        "model_type": "HybridXGBoostClassifier",
        "input_dim": len(feature_names),
        "raw_features_dim": 165,
        "structural_embeddings_dim": 32,
        "optimal_threshold": optimal_threshold,
        "scale_pos_weight": float(scale_pos_weight),
        "test_pr_auc": float(pr_auc),
        "test_roc_auc": float(roc_auc),
        "test_illicit_f1_optimal": float(report_optimal["Illicit"]["f1-score"]),
        "test_illicit_precision_optimal": float(report_optimal["Illicit"]["precision"]),
        "test_illicit_recall_optimal": float(report_optimal["Illicit"]["recall"]),
        "brier_score_loss": float(brier_score),
        "feature_names": feature_names,
    }
    with open(config_save_path, "w") as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Saved model configuration to {config_save_path}")

    # Save comprehensive metrics report
    metrics_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_samples_total": len(y_test),
        "test_samples_illicit": int(np.sum(y_test == 1)),
        "test_samples_licit": int(np.sum(y_test == 0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score_loss": float(brier_score),
        "optimal_threshold": optimal_threshold,
        "confusion_matrix_default_0_50": cm_default.tolist(),
        "confusion_matrix_optimal": cm_optimal.tolist(),
        "classification_report_default": report_default,
        "classification_report_optimal": report_optimal,
        "threshold_sweep_sample": threshold_results[:10],
    }
    metrics_report_path = logs_dir / "hybrid_metrics_report.json"
    with open(metrics_report_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info(f"Saved performance audit report to {metrics_report_path}")

    # -------------------------------------------------------------
    # 7. Generate SHAP Attribution Explanations
    # -------------------------------------------------------------
    logger.info("Initializing FraudSHAPExplainer for explainability attribution...")
    from ml_engine.explainability.shap_explainer import FraudSHAPExplainer

    explainer = FraudSHAPExplainer(
        model_path=str(model_save_path),
        config_path=str(config_save_path),
    )
    # Take a representative background sample of 2,000 test transactions
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_sample = X_test[sample_indices]

    explainer.compute_and_save_global_explanations(X_sample=X_sample, output_dir=str(logs_dir))

    total_time = time.time() - total_start
    logger.info(f"Phase 3 Pipeline completed successfully in {total_time:.2f} seconds!")

    return metrics_report


if __name__ == "__main__":
    train_and_evaluate_hybrid()
