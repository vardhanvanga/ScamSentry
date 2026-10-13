"""
Real-Time Transaction Switch Emulator & Streaming Simulator
============================================================
Emulates a live payment switch transaction stream at configurable TPS rates,
with stratified traffic mixture (90% routine / 7% borderline / 3% syndicate).

Phase 5 — Pillar 3
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SwitchStreamSimulator:
    """
    Asynchronous transaction switch emulator that streams test transactions
    at configurable TPS rates with realistic traffic distribution.

    Traffic mixture (per real-world AML monitoring baselines):
      - 90% ROUTINE      : Class 0 / Licit transactions
      -  7% BORDERLINE   : Mid-risk velocity surge transactions
      -  3% SYNDICATE    : Class 1 / Illicit coordinated flows
    """

    _TRAFFIC_WEIGHTS = [0.90, 0.07, 0.03]
    _TRAFFIC_TYPES   = ["ROUTINE", "BORDERLINE", "SYNDICATE"]

    def __init__(self, pipeline, tps: int = 10):
        """
        Args:
            pipeline: Loaded RealTimeInferencePipeline instance.
            tps: Transactions per second to emit (default 10; supports up to 50).
        """
        from backend.app.engine.streaming_pipeline import RealTimeInferencePipeline  # noqa: F401 type hint
        self.pipeline = pipeline
        self.tps = max(1, min(tps, 200))  # clamp to sane range
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._running: bool = False

        # Build stratified node pools from test mask
        self.licit_nodes: np.ndarray
        self.borderline_nodes: np.ndarray
        self.illicit_nodes: np.ndarray
        self._build_node_pools()

        logger.info(
            f"SwitchStreamSimulator ready | TPS={self.tps} | "
            f"Pools: ROUTINE={len(self.licit_nodes)}, "
            f"BORDERLINE={len(self.borderline_nodes)}, "
            f"SYNDICATE={len(self.illicit_nodes)}"
        )

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _build_node_pools(self) -> None:
        """
        Build stratified node index pools from the graph test partition.
        Uses ground-truth labels (y) when available; falls back to random split.
        """
        graph = self.pipeline.graph
        test_mask = graph.test_mask

        test_indices = torch.where(test_mask)[0].cpu().numpy()
        np.random.seed(42)

        if hasattr(graph, "y") and graph.y is not None:
            y_test = graph.y[test_mask].cpu().numpy()
            licit_mask   = (y_test == 0)
            illicit_mask = (y_test == 1)

            self.licit_nodes   = test_indices[licit_mask]
            self.illicit_nodes = test_indices[illicit_mask]

            # Borderline: random 15% sample from licit pool (velocity-surge heuristic)
            borderline_size = max(1, int(len(self.licit_nodes) * 0.15))
            self.borderline_nodes = np.random.choice(
                self.licit_nodes, size=borderline_size, replace=False
            )

        else:
            # No labels available — split 80 / 15 / 5
            n = len(test_indices)
            shuffled = np.random.permutation(test_indices)
            cut1 = int(n * 0.80)
            cut2 = int(n * 0.95)
            self.licit_nodes      = shuffled[:cut1]
            self.borderline_nodes = shuffled[cut1:cut2]
            self.illicit_nodes    = shuffled[cut2:]

        # Safety: ensure no pool is empty
        if len(self.illicit_nodes) == 0:
            self.illicit_nodes = np.random.choice(test_indices, size=1)
        if len(self.borderline_nodes) == 0:
            self.borderline_nodes = np.random.choice(test_indices, size=1)

    # ------------------------------------------------------------------
    # Callback management
    # ------------------------------------------------------------------

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callable to be invoked with each transaction result dict.

        Args:
            callback: Function accepting a single result dict argument.
        """
        self._callbacks.append(callback)
        logger.debug(f"Registered callback #{len(self._callbacks)}: {callback}")

    def _fire_callbacks(self, result: Dict[str, Any]) -> None:
        """Invoke all registered callbacks with the given result."""
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as exc:
                logger.warning(f"Callback error: {exc}")

    # ------------------------------------------------------------------
    # Transaction sampling
    # ------------------------------------------------------------------

    def _sample_next_transaction(self) -> tuple:
        """
        Weighted random selection of next transaction node and traffic type.

        Returns:
            (node_idx: int, traffic_type: str)
        """
        traffic_type = random.choices(self._TRAFFIC_TYPES, weights=self._TRAFFIC_WEIGHTS, k=1)[0]

        pool_map = {
            "ROUTINE":    self.licit_nodes,
            "BORDERLINE": self.borderline_nodes,
            "SYNDICATE":  self.illicit_nodes,
        }
        pool = pool_map[traffic_type]
        node_idx = int(np.random.choice(pool))
        return node_idx, traffic_type

    # ------------------------------------------------------------------
    # Async transaction emission
    # ------------------------------------------------------------------

    async def _emit_transaction(self, node_idx: int, traffic_type: str) -> Dict[str, Any]:
        """
        Run inference on a single node in a thread pool executor (non-blocking).

        Args:
            node_idx: PyG graph node index.
            traffic_type: Emulated traffic category string.

        Returns:
            Result dict with inference output and metadata.
        """
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, self.pipeline.process_incoming_transaction, node_idx
        )
        return {
            "node_idx": node_idx,
            "traffic_type": traffic_type,
            "tx_id": resp.tx_id,
            "risk_score": resp.risk_score,
            "risk_tier": resp.risk_tier,
            "action": resp.recommended_action,
            "reason": resp.reason_codes[0] if resp.reason_codes else "",
            "latency_ms": resp.latency_ms,
            "sla_compliant": resp.switch_compliance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Main streaming loop
    # ------------------------------------------------------------------

    async def stream(
        self,
        duration_seconds: float = 30.0,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Async streaming loop: emits transactions at self.tps rate.

        Args:
            duration_seconds: Total streaming window in seconds.
            on_result: Optional additional callback per result.

        Returns:
            List of all result dicts produced during the session.
        """
        self._running = True
        results: List[Dict[str, Any]] = []
        interval = 1.0 / self.tps
        deadline = time.perf_counter() + duration_seconds
        tx_count = 0

        logger.info(
            f"SwitchStreamSimulator STREAM START | TPS={self.tps} | "
            f"Duration={duration_seconds}s"
        )

        try:
            while self._running and time.perf_counter() < deadline:
                t_iter_start = time.perf_counter()
                node_idx, traffic_type = self._sample_next_transaction()

                try:
                    result = await self._emit_transaction(node_idx, traffic_type)
                except Exception as exc:
                    logger.warning(f"Transaction emit error (node {node_idx}): {exc}")
                    result = {
                        "node_idx": node_idx,
                        "traffic_type": traffic_type,
                        "error": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                results.append(result)
                self._fire_callbacks(result)
                if on_result:
                    try:
                        on_result(result)
                    except Exception as exc:
                        logger.warning(f"on_result callback error: {exc}")

                tx_count += 1
                elapsed = time.perf_counter() - t_iter_start
                sleep_time = max(0.0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        finally:
            self._running = False
            logger.info(
                f"SwitchStreamSimulator STREAM END | Transactions: {tx_count}"
            )

        return results

    def stop(self) -> None:
        """Signal the running stream to stop gracefully."""
        self._running = False

    def run_sync(
        self,
        duration_seconds: float = 10.0,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for non-async callers.

        Args:
            duration_seconds: Streaming window in seconds.
            on_result: Optional per-result callback.

        Returns:
            List of all result dicts.
        """
        return asyncio.run(self.stream(duration_seconds=duration_seconds, on_result=on_result))

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def get_traffic_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute summary statistics over a set of streamed results.

        Args:
            results: List of result dicts from stream() or run_sync().

        Returns:
            Summary dict with counts, latency percentiles, and SLA compliance rate.
        """
        if not results:
            return {"total_transactions": 0}

        by_traffic: Dict[str, int] = {}
        by_action: Dict[str, int] = {}
        latencies: List[float] = []
        sla_pass = 0

        for r in results:
            if "error" in r:
                continue
            by_traffic[r.get("traffic_type", "UNKNOWN")] = by_traffic.get(r.get("traffic_type", "UNKNOWN"), 0) + 1
            by_action[r.get("action", "UNKNOWN")] = by_action.get(r.get("action", "UNKNOWN"), 0) + 1
            lat = r.get("latency_ms", 0.0)
            latencies.append(lat)
            if r.get("sla_compliant", False):
                sla_pass += 1

        lat_arr = np.array(latencies) if latencies else np.array([0.0])

        return {
            "total_transactions": len(results),
            "error_count": sum(1 for r in results if "error" in r),
            "by_traffic_type": by_traffic,
            "by_action": by_action,
            "avg_latency_ms": round(float(np.mean(lat_arr)), 2),
            "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 2),
            "sla_pass_rate_pct": round((sla_pass / max(len(latencies), 1)) * 100.0, 1),
        }
