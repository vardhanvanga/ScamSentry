import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from .env if present
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
dotenv_path = ROOT_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)


class Settings:
    """Enterprise application configuration settings."""

    # Project Information
    APP_NAME: str = os.getenv("APP_NAME", "ScamSentry")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Directory Paths
    BASE_DIR: Path = ROOT_DIR
    DATA_RAW_DIR: Path = ROOT_DIR / os.getenv("DATA_RAW_DIR", "data/raw")
    DATA_PROCESSED_DIR: Path = ROOT_DIR / os.getenv("DATA_PROCESSED_DIR", "data/processed")
    PROCESSED_GRAPH_FILE: Path = DATA_PROCESSED_DIR / os.getenv(
        "PROCESSED_GRAPH_FILENAME", "elliptic_pyg_graph.pt"
    )

    ARTIFACTS_DIR: Path = ROOT_DIR / os.getenv("ARTIFACTS_DIR", "artifacts")
    SAVED_MODELS_DIR: Path = ROOT_DIR / os.getenv("SAVED_MODELS_DIR", "artifacts/saved_models")
    LOGS_DIR: Path = ROOT_DIR / os.getenv("LOGS_DIR", "artifacts/logs")

    # Temporal & GNN Settings
    TEMPORAL_TRAIN_CUTOFF: int = int(os.getenv("TEMPORAL_TRAIN_CUTOFF", "34"))
    TEMPORAL_TOTAL_TIMESTEPS: int = int(os.getenv("TEMPORAL_TOTAL_TIMESTEPS", "49"))

    # Policy & Fraud Engine Settings
    FRAUD_RISK_THRESHOLD: float = float(os.getenv("FRAUD_RISK_THRESHOLD", "0.75"))
    AUTOMATIC_FREEZE_ENABLED: bool = (
        os.getenv("AUTOMATIC_FREEZE_ENABLED", "True").lower() in ("true", "1", "yes")
    )


settings = Settings()
