"""
ScamSentry STAR-Fraud Engine — Enterprise FastAPI Gateway
==========================================================
Phase 5 / Pillar 1 — Production-grade REST API wrapping RealTimeInferencePipeline.

Endpoints
---------
POST /api/v1/switch/evaluate    — Real-time transaction risk evaluation
GET  /api/v1/switch/health      — System health & loaded artifact versions
GET  /api/v1/switch/metrics     — Running latency percentiles & triage statistics

Startup
-------
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── project root on sys.path ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.app.core.config import settings
from backend.app.engine.compliance import DomainFeatureMapper, RegulatoryAuditLogger

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Globals (populated during lifespan startup) ──────────────────────────────
streaming_pipeline = None          # RealTimeInferencePipeline
domain_mapper      = DomainFeatureMapper()
audit_logger       = RegulatoryAuditLogger()

# In-memory circular buffer for rolling latency statistics (last 10 000 requests)
_latency_store: deque = deque(maxlen=10_000)
_action_counts: Dict[str, int] = {"APPROVE": 0, "STEP_UP_MFA": 0, "FLAG_HIGH_RISK": 0}
_total_requests: int = 0
_sla_violations: int = 0


# ── Application lifespan (replaces deprecated @app.on_event) ─────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML pipeline once at startup; release on shutdown."""
    global streaming_pipeline
    logger.info("═" * 72)
    logger.info("  ScamSentry STAR-Fraud Engine — FastAPI Gateway starting up…  ")
    logger.info("═" * 72)
    try:
        from backend.app.engine.streaming_pipeline import RealTimeInferencePipeline
        streaming_pipeline = RealTimeInferencePipeline()
        logger.info("✅  RealTimeInferencePipeline loaded and warm.")
    except Exception as exc:
        logger.critical(f"❌  Pipeline init failed: {exc}")
        streaming_pipeline = None

    yield  # ── application is live ──────────────────────────────────────────

    logger.info("ScamSentry gateway shutting down.")


# ── FastAPI application ──────────────────────────────────────────────────────
app = FastAPI(
    title="ScamSentry STAR-Fraud Engine — Enterprise API",
    description=(
        "Production-grade real-time financial fraud detection gateway. "
        "Combines a Graph Temporal Transformer with hybrid XGBoost and "
        "SHAP-based explainability under a sub-100 ms payment-switch SLA."
    ),
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "switch",  "description": "Payment switch evaluation & streaming endpoints"},
        {"name": "monitor", "description": "Health probes and production telemetry"},
    ],
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas (Pydantic V2) ─────────────────────────────────

class TransactionPayload(BaseModel):
    """Typed incoming payment-switch transaction payload."""

    tx_id: str = Field(
        ...,
        pattern=r"^[A-Za-z0-9_\-]{1,64}$",
        description="Alphanumeric financial transaction identifier (max 64 chars)",
        examples=["TXN-2026-001ABC"],
    )
    source_account_id: str = Field(
        ...,
        description="Source account or UPI identifier",
        examples=["ACC_105"],
    )
    target_node_idx: int = Field(
        ...,
        ge=0,
        description="Target graph node index (must be ≥ 0)",
        examples=[168901],
    )
    amount: float = Field(
        ...,
        gt=0.0,
        description="Transaction amount in INR (must be > 0)",
        examples=[1450.75],
    )
    timestamp: int = Field(
        ...,
        description="Unix epoch timestamp of the transaction",
        examples=[1727000000],
    )
    device_fingerprint: Optional[str] = Field(
        None,
        description="Optional device fingerprint for step-up auth context",
    )

    @field_validator("tx_id")
    @classmethod
    def tx_id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("tx_id must not be blank")
        return v.strip()


class ExplainabilityEntry(BaseModel):
    """Single SHAP feature attribution entry."""
    feature_name: str
    domain_label: str
    attribution_weight: float
    feature_value: float
    direction: str  # "RISK_ELEVATING" | "RISK_MITIGATING"


class SwitchDecisionResponse(BaseModel):
    """Regulatory-compliant switch decision with full explainability metadata."""

    transaction_id: str = Field(description="Internal graph transaction ID")
    risk_score: float = Field(ge=0.0, le=1.0, description="Fraud probability [0, 1]")
    risk_tier: str = Field(description="LOW_RISK | MEDIUM_RISK | HIGH_RISK")
    action: str = Field(description="APPROVE | STEP_UP_MFA | FLAG_HIGH_RISK")
    explainability_summary: List[ExplainabilityEntry] = Field(
        description="SHAP feature attributions with banking-domain labels"
    )
    processing_latency_ms: float = Field(description="End-to-end inference latency in ms")
    sla_compliant: bool = Field(description="True if processing_latency_ms < 100.0 ms")
    trace_id: str = Field(description="UUID4 distributed trace identifier")


class HealthResponse(BaseModel):
    """System health and loaded artifact inventory."""
    status: str
    pipeline_loaded: bool
    torch_device: str
    model_artifact_versions: Dict[str, str]
    memory_utilization_pct: float
    graph_nodes: Optional[int]
    graph_edges: Optional[int]
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Running production telemetry for SRE dashboards."""
    total_requests: int
    action_counts: Dict[str, int]
    sla_violation_count: int
    sla_violation_rate_pct: float
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    latency_p99_ms: Optional[float]
    avg_latency_ms: Optional[float]


# ── Helper utilities ─────────────────────────────────────────────────────────

_server_start = time.perf_counter()

import numpy as np  # noqa: E402  (after stdlib/fastapi imports for clarity)


def _percentile(data: list, p: float) -> Optional[float]:
    if not data:
        return None
    return round(float(np.percentile(data, p)), 2)


def _get_model_versions() -> Dict[str, str]:
    """Return model artifact file names as version strings."""
    versions: Dict[str, str] = {}
    for key, fname in [
        ("graph_transformer", "graph_transformer_best.pt"),
        ("hybrid_xgboost",   "hybrid_xgboost_best.json"),
        ("model_config",     "model_config.json"),
    ]:
        path = settings.SAVED_MODELS_DIR / fname
        if path.exists():
            mtime = path.stat().st_mtime
            versions[key] = f"{fname} (mtime={int(mtime)})"
        else:
            versions[key] = "NOT_FOUND"
    return versions


# ── Routes ───────────────────────────────────────────────────────────────────

from fastapi import APIRouter  # noqa: E402

switch_router = APIRouter(prefix="/api/v1/switch", tags=["switch"])


@switch_router.post(
    "/evaluate",
    response_model=SwitchDecisionResponse,
    summary="Real-time transaction risk evaluation",
    description=(
        "Submit an incoming payment-switch transaction for real-time fraud scoring. "
        "Returns a typed risk decision with SHAP-powered explainability and RBI "
        "FREE-AI compliant adverse action reason codes. SLA target: < 100 ms."
    ),
)
async def evaluate_transaction(payload: TransactionPayload, request: Request):
    """
    POST /api/v1/switch/evaluate

    Calls RealTimeInferencePipeline.process_incoming_transaction() and enriches
    the result with DomainFeatureMapper explainability and UUID4 trace ID.
    Logs all FLAG_HIGH_RISK and STEP_UP_MFA decisions to the regulatory audit trail.
    """
    global _total_requests, _sla_violations

    if streaming_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="ML pipeline not loaded. Check server logs for initialization errors.",
        )

    trace_id = str(uuid.uuid4())
    _total_requests += 1

    # ── Validate node index bounds ───────────────────────────────────────────
    num_nodes = streaming_pipeline.graph.num_nodes
    if payload.target_node_idx >= num_nodes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"target_node_idx {payload.target_node_idx} exceeds graph size "
                f"({num_nodes} nodes). Valid range: 0 – {num_nodes - 1}."
            ),
        )

    # ── Run ML inference ─────────────────────────────────────────────────────
    try:
        decision = streaming_pipeline.process_incoming_transaction(payload.target_node_idx)
    except Exception as exc:
        logger.error(f"[Trace {trace_id}] Inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {exc}")

    # ── SHAP Explainability enrichment ───────────────────────────────────────
    explainability_summary: List[ExplainabilityEntry] = []
    try:
        # Re-extract SHAP for explainability summary (runs only for MFA/FLAG)
        if decision.recommended_action in {"STEP_UP_MFA", "FLAG_HIGH_RISK"}:
            sub_x, sub_edge_index, target_pos = streaming_pipeline._sample_2hop_subgraph(
                payload.target_node_idx
            )
            import torch as _torch
            with _torch.no_grad():
                sub_embs, _ = streaming_pipeline.gtt_model(sub_x, sub_edge_index)
                target_emb = sub_embs[target_pos: target_pos + 1].cpu().numpy()

            import numpy as _np
            raw_feat = streaming_pipeline.graph.x[
                payload.target_node_idx: payload.target_node_idx + 1
            ].cpu().numpy()
            hybrid_vec = _np.hstack([raw_feat, target_emb]).astype(_np.float32)
            shap_result = streaming_pipeline.explainer.explain_single_transaction(hybrid_vec[0])
            raw_summary = domain_mapper.format_explainability_summary(shap_result)
            explainability_summary = [ExplainabilityEntry(**e) for e in raw_summary]
    except Exception as exc:
        logger.warning(f"[Trace {trace_id}] SHAP enrichment skipped: {exc}")

    # Fall back to reason_codes if no SHAP entries
    if not explainability_summary:
        explainability_summary = [
            ExplainabilityEntry(
                feature_name="policy_engine",
                domain_label=rc,
                attribution_weight=0.0,
                feature_value=0.0,
                direction="RISK_ELEVATING" if decision.risk_score >= 0.40 else "RISK_MITIGATING",
            )
            for rc in decision.reason_codes[:5]
        ]

    # ── Metrics update ───────────────────────────────────────────────────────
    _latency_store.append(decision.latency_ms)
    _action_counts[decision.recommended_action] = (
        _action_counts.get(decision.recommended_action, 0) + 1
    )
    if not decision.switch_compliance:
        _sla_violations += 1

    # ── Regulatory audit logging ─────────────────────────────────────────────
    audit_logger.log_flagged_transaction(
        trace_id=trace_id,
        payload=payload.model_dump(),
        decision=decision,
        shap_summary=[e.model_dump() for e in explainability_summary],
    )

    return SwitchDecisionResponse(
        transaction_id=decision.tx_id,
        risk_score=decision.risk_score,
        risk_tier=decision.risk_tier,
        action=decision.recommended_action,
        explainability_summary=explainability_summary,
        processing_latency_ms=decision.latency_ms,
        sla_compliant=decision.switch_compliance,
        trace_id=trace_id,
    )


@switch_router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health and artifact inventory",
    tags=["monitor"],
)
async def health_check():
    """
    GET /api/v1/switch/health

    Reports pipeline readiness, PyTorch device, memory utilization,
    and loaded model artifact version metadata.
    """
    loaded = streaming_pipeline is not None
    device_str = "N/A"
    graph_nodes = graph_edges = None

    if loaded:
        device_str = str(next(streaming_pipeline.gtt_model.parameters()).device)
        graph_nodes = int(streaming_pipeline.graph.num_nodes)
        graph_edges = int(streaming_pipeline.graph.num_edges)

    mem = psutil.virtual_memory()
    mem_pct = round(mem.percent, 1)

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        pipeline_loaded=loaded,
        torch_device=device_str,
        model_artifact_versions=_get_model_versions(),
        memory_utilization_pct=mem_pct,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        uptime_seconds=round(time.perf_counter() - _server_start, 1),
    )


@switch_router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Production telemetry and SLA statistics",
    tags=["monitor"],
)
async def get_metrics():
    """
    GET /api/v1/switch/metrics

    Exposes running statistics: total requests, p50/p95/p99 latencies,
    triage action counts, and SLA violation rate.
    """
    lat_list = list(_latency_store)
    sla_rate = (
        round((_sla_violations / _total_requests) * 100.0, 2)
        if _total_requests > 0 else 0.0
    )

    return MetricsResponse(
        total_requests=_total_requests,
        action_counts=dict(_action_counts),
        sla_violation_count=_sla_violations,
        sla_violation_rate_pct=sla_rate,
        latency_p50_ms=_percentile(lat_list, 50),
        latency_p95_ms=_percentile(lat_list, 95),
        latency_p99_ms=_percentile(lat_list, 99),
        avg_latency_ms=round(float(np.mean(lat_list)), 2) if lat_list else None,
    )


app.include_router(switch_router)


# ── Root endpoint ─────────────────────────────────────────────────────────────
@app.get("/", tags=["monitor"])
async def root():
    return {
        "engine":  "ScamSentry STAR-Fraud Engine",
        "phase":   "Phase 5 — Enterprise Production Gateway",
        "status":  "online",
        "docs":    "/docs",
        "redoc":   "/redoc",
        "evaluate": "POST /api/v1/switch/evaluate",
        "health":  "GET /api/v1/switch/health",
        "metrics": "GET /api/v1/switch/metrics",
    }


# ── Legacy v1 router (backward-compat with Phase 1 callers) ──────────────────
from backend.app.api.routes import router as _legacy_router  # noqa: E402
app.include_router(_legacy_router, prefix="/api/v1/legacy", tags=["legacy"])


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
