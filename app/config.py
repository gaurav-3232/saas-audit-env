"""Application configuration loaded from environment variables."""

import os


class Config:
    """Central configuration for SaaSAuditEnv."""

    # Flask
    SECRET_KEY: str = os.getenv("SECRET_KEY", "saas-audit-env-dev-secret")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "7860"))

    # Episode limits
    DEFAULT_MAX_STEPS: int = int(os.getenv("DEFAULT_MAX_STEPS", "20"))


config = Config()
