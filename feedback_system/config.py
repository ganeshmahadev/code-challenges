"""Configuration management for the feedback system."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")
    AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "5"))
    AI_PROVIDER_ENABLED = os.getenv("AI_PROVIDER_ENABLED", "true").lower() == "true"

    # API Configuration
    API_KEY = os.getenv("API_KEY", "test-api-key-12345")

    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./feedback.db")

    # Cache Configuration
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    # Alert Configuration
    ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
    ALERT_ENABLED = os.getenv("ALERT_ENABLED", "false").lower() == "true"

    # Topics Configuration
    SUPPORTED_TOPICS = [
        "billing",
        "product_features",
        "technical_issue",
        "customer_support",
        "delivery",
        "general"
    ]

    # Alert triggers
    ALERT_KEYWORDS = [
        "cancel", "refund", "lawsuit", "lawyer", "legal",
        "urgent", "emergency", "critical", "immediately"
    ]


config = Config()
