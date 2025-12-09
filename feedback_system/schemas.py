"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class FeedbackRequest(BaseModel):
    """Request schema for feedback submission."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "I love the new features but the billing page is confusing."
            }
        }
    )

    text: str = Field(..., min_length=1, max_length=5000, description="Customer feedback text")


class FeedbackResponse(BaseModel):
    """Response schema for feedback analysis."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "sentiment": "mixed",
                "topic": "billing",
                "confidence_score": "high",
                "alert_triggered": False,
                "processing_method": "ai",
                "created_at": "2025-12-06T10:30:00"
            }
        }
    )

    id: int
    sentiment: str = Field(..., description="Sentiment classification: positive, negative, or neutral")
    topic: str = Field(..., description="Main topic identified in the feedback")
    confidence_score: Optional[str] = Field(None, description="Confidence level: high, medium, or low")
    alert_triggered: bool = Field(..., description="Whether this feedback triggered an alert")
    processing_method: str = Field(..., description="How the feedback was processed: ai or fallback")
    created_at: str


class AnalysisResult(BaseModel):
    """Internal schema for analysis results."""

    sentiment: str
    topic: str
    confidence_score: str = "medium"
    alert_triggered: bool = False
    processing_method: str = "ai"
