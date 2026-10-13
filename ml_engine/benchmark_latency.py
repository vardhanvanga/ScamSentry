import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.engine.streaming_pipeline import RealTimeInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BenchmarkLatency")


def run_latency_benchmark(num_samples: int = 100, num_warmup: int = 5) -> Dict[str, Any]:
    """
    Executes Method D Latency Benchmarking across simulated streaming transactions
    drawn from the held-out temporal test set.

    Args:
        num_samples: Number of timed streaming transaction evaluations.
        num_warmup: Number of untimed warmup iterations to prime JIT & CPU caches.

    Returns:
        Dictionary of latency statistics and SLA compliance results.
    """
    logger.info("================================================================================")
    logger.info("   SCAMSENTRY (STAR-FRAUD ENGINE) — PHASE 4 REAL-TIME LATENCY BENCHMARK        ")
    logger.info("================================================================================")

    # 1. Initialize Real-Time Pipeline
    pipeline = RealTimeInferencePipeline()

    # 2. Extract Test Nodes (from held-out temporal partition)
    test_node_indices = torch.where(pipeline.graph.test_mask)[0].cpu().numpy()
    np.random.seed(42)
    selected_indices = np.random.choice(test_node_indices, size=num_samples + num_warmup, replace=False)

    warmup_nodes = selected_indices[:num_warmup]
    benchmark_nodes = selected_indices[num_warmup:]

    # 3. Warm-up Phase
    logger.info(f"Executing {num_warmup} warm-up transactions to prime runtime caches...")
    for idx in warmup_nodes:
        _ = pipeline.process_incoming_transaction(int(idx))
    logger.info("Warm-up complete.")

    # 4. Benchmarking Loop
    logger.info(f"Benchmarking {num_samples} simulated streaming transactions (Method D)...")
    latencies: List[float] = []
    actions: Dict[str, int] = {"APPROVE": 0, "STEP_UP_MFA": 0, "FLAG_HIGH_RISK": 0}
    decisions_log: List[Dict[str, Any]] = []

    t_bench_start = time.perf_counter()
    for i, node_idx in enumerate(benchmark_nodes):
        resp = pipeline.process_incoming_transaction(int(node_idx))
        latencies.append(resp.latency_ms)
        actions[resp.recommended_action] = actions.get(resp.recommended_action, 0) + 1

        decisions_log.append({
            "node_idx": int(node_idx),
            "tx_id": resp.tx_id,
            "risk_score": resp.risk_score,
            "risk_tier": resp.risk_tier,
            "recommended_action": resp.recommended_action,
            "latency_ms": resp.latency_ms,
            "switch_compliance": resp.switch_compliance,
            "top_reason": resp.reason_codes[0] if resp.reason_codes else "",
        })

    total_bench_duration = time.perf_counter() - t_bench_start
    lat_arr = np.array(latencies)

    # 5. Compute Latency Statistics
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_lat = float(np.mean(lat_arr))
    std_lat = float(np.std(lat_arr))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))

    # SLA Pass Rate (< 100.0 ms)
    compliant_count = int(np.sum(lat_arr < 100.0))
    sla_pass_rate = float((compliant_count / num_samples) * 100.0)

    # 6. Terminal Dashboard Display
    logger.info("================================================================================")
    logger.info("              PAYMENT SWITCH REAL-TIME SLA BENCHMARK REPORT                     ")
    logger.info("================================================================================")
    logger.info(f"Total Transactions Evaluated:       {num_samples}")
    logger.info(f"Target Switch SLA Limit:            < 100.00 ms")
    logger.info(f"SLA Compliance Pass Rate:           {sla_pass_rate:.1f}% ({compliant_count}/{num_samples})")
    logger.info("-" * 80)
    logger.info("LATENCY PERCENTILE BREAKDOWN:")
    logger.info(f"  • Median Latency (p50):           {p50:.2f} ms")
    logger.info(f"  • 95th Percentile (p95):          {p95:.2f} ms")
    logger.info(f"  • 99th Percentile (p99):          {p99:.2f} ms")
    logger.info(f"  • Mean Latency:                   {mean_lat:.2f} ms (+/- {std_lat:.2f} ms)")
    logger.info(f"  • Range (Min -> Max):             {min_lat:.2f} ms -> {max_lat:.2f} ms")
    logger.info("-" * 80)
    logger.info("3-TIER POLICY DECISION DISTRIBUTION:")
    logger.info(f"  • Frictionless APPROVE:           {actions.get('APPROVE', 0)} ({actions.get('APPROVE', 0)/num_samples*100:.1f}%)")
    logger.info(f"  • Dynamic STEP_UP_MFA:            {actions.get('STEP_UP_MFA', 0)} ({actions.get('STEP_UP_MFA', 0)/num_samples*100:.1f}%)")
    logger.info(f"  • Immediate FLAG_HIGH_RISK:       {actions.get('FLAG_HIGH_RISK', 0)} ({actions.get('FLAG_HIGH_RISK', 0)/num_samples*100:.1f}%)")
    logger.info(f"  • Total Flagged Syndicate Nodes:  {len(pipeline.triaging_engine.flagged_syndicate_nodes)}")
    logger.info("================================================================================")

    # 7. Persist Artifacts
    logs_dir = ROOT_DIR / "artifacts" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_file = logs_dir / "latency_benchmark_report.json"

    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": num_samples,
        "num_warmup": num_warmup,
        "total_benchmark_duration_sec": round(total_bench_duration, 2),
        "sla_limit_ms": 100.0,
        "sla_pass_rate_pct": sla_pass_rate,
        "latency_metrics_ms": {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "mean": round(mean_lat, 2),
            "std": round(std_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2),
        },
        "decision_distribution": actions,
        "flagged_syndicate_count": len(pipeline.triaging_engine.flagged_syndicate_nodes),
        "raw_latencies_ms": [round(x, 2) for x in latencies],
        "sample_decisions": decisions_log[:10],
    }

    with open(report_file, "w") as f:
        json.dump(benchmark_report, f, indent=2)
    logger.info(f"Saved benchmark report to {report_file}")

    return benchmark_report


if __name__ == "__main__":
    run_latency_benchmark(num_samples=100, num_warmup=5)
