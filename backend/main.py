"""
ScamSentry — Backend Entrypoint
================================
Delegates to the Phase 5 enterprise app in backend.app.main.
"""
import uvicorn
from backend.app.main import app  # noqa: F401 — re-export for uvicorn
from backend.app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
