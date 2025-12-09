"""Database models for feedback storage."""
from datetime import datetime, UTC
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Feedback(Base):
    """Feedback database model."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False)  # positive, negative, neutral
    topic = Column(String(50), nullable=False)
    confidence_score = Column(String(20), nullable=True)  # high, medium, low
    alert_triggered = Column(Boolean, default=False)
    processing_method = Column(String(20), nullable=False)  # ai or fallback
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "sentiment": self.sentiment,
            "topic": self.topic,
            "confidence_score": self.confidence_score,
            "alert_triggered": self.alert_triggered,
            "processing_method": self.processing_method,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
