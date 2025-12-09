#!/usr/bin/env python3
"""
This file consolidates all tests:
- Unit tests for analyzers (AI, ML fallback)
- Integration tests for database operations
- Alert system tests (sentiment-based alerts)
- Slack webhook integration tests
- End-to-end workflow tests
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from config import config
from schemas import AnalysisResult
from database import init_db, get_db_session, save_feedback
from ai_analyzer import AIAnalyzer
from ml_fallback_analyzer import MLFallbackAnalyzer
from alerting import AlertService


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
async def db_session():
    """Initialize database for testing."""
    await init_db()
    async with get_db_session() as session:
        yield session

# ============================================================================
# UNIT TESTS - AI ANALYZER (OpenAI API)
# ============================================================================

class TestOpenAIAPIIntegration:
    """Tests for OpenAI API integration and analysis."""

    @pytest.mark.asyncio
    async def test_ai_analyzer_with_real_api(self):
        """Test AI analyzer with real OpenAI API (if configured)."""
        if not config.AI_PROVIDER_ENABLED or not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API not configured")

        analyzer = AIAnalyzer()
        feedback_text = "I love this product! It's amazing and works perfectly."

        result = await analyzer.analyze(feedback_text)

        assert result.sentiment == "positive"
        assert result.processing_method == "ai"
        assert result.confidence_score in ["high", "medium", "low"]
        assert result.alert_triggered is False

    @pytest.mark.asyncio
    async def test_ai_analyzer_negative_feedback(self):
        """Test AI analyzer correctly identifies negative feedback."""
        if not config.AI_PROVIDER_ENABLED or not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API not configured")

        analyzer = AIAnalyzer()
        feedback_text = "This is terrible! I hate this product and want a refund."

        result = await analyzer.analyze(feedback_text)

        assert result.sentiment == "negative"
        assert result.processing_method == "ai"
        # Alert only triggered if confidence is high or medium (not low)
        if result.confidence_score in ["high", "medium"]:
            assert result.alert_triggered is True
        else:
            assert result.alert_triggered is False

    @pytest.mark.asyncio
    async def test_ai_analyzer_mixed_sentiment(self):
        """Test AI analyzer handles mixed sentiment."""
        if not config.AI_PROVIDER_ENABLED or not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API not configured")

        analyzer = AIAnalyzer()
        feedback_text = "I love the features but the billing is completely broken!"

        result = await analyzer.analyze(feedback_text)

        assert result.sentiment in ["mixed", "negative"]
        assert result.processing_method == "ai"
        # Alert only triggered if confidence is high or medium (not low)
        if result.confidence_score in ["high", "medium"]:
            assert result.alert_triggered is True
        else:
            assert result.alert_triggered is False

    @pytest.mark.asyncio
    async def test_ai_analyzer_topic_classification(self):
        """Test AI analyzer correctly classifies topics."""
        if not config.AI_PROVIDER_ENABLED or not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API not configured")

        analyzer = AIAnalyzer()
        feedback_text = "I was charged twice for my subscription this month!"

        result = await analyzer.analyze(feedback_text)

        assert result.topic == "billing"
        assert result.processing_method == "ai"

    @pytest.mark.asyncio
    async def test_ai_analyzer_timeout_handling(self):
        """Test that AI analyzer respects timeout settings."""
        if not config.AI_PROVIDER_ENABLED or not config.OPENAI_API_KEY:
            pytest.skip("OpenAI API not configured")

        # This test verifies timeout mechanism exists, actual timeout may not trigger
        analyzer = AIAnalyzer()
        assert analyzer.timeout == config.AI_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_ai_fallback_on_error(self):
        """Test that system falls back to ML when AI fails."""
        # Temporarily disable AI to test fallback
        original_enabled = config.AI_PROVIDER_ENABLED
        config.AI_PROVIDER_ENABLED = False

        analyzer = AIAnalyzer()
        fallback_analyzer = MLFallbackAnalyzer()

        feedback_text = "This product is terrible"

        # AI should fail, system should use ML fallback
        try:
            result = await analyzer.analyze(feedback_text)
            # Should not reach here
            assert False, "Should have raised exception"
        except Exception:
            # Expected - AI should fail when disabled
            result = fallback_analyzer.analyze(feedback_text)
            assert result.sentiment == "negative"
            assert result.processing_method == "ml_fallback"

        # Restore original config
        config.AI_PROVIDER_ENABLED = original_enabled


class TestAIResponseParsing:
    """Tests for AI response parsing and validation."""

    @pytest.mark.asyncio
    async def test_parse_valid_json_response(self):
        """Should correctly parse valid JSON from AI."""
        analyzer = AIAnalyzer()
        response = '{"sentiment": "positive", "topic": "billing", "confidence": "high", "needs_alert": false}'

        result = analyzer._parse_ai_response(response)

        assert result["sentiment"] == "positive"
        assert result["topic"] == "billing"
        assert result["confidence"] == "high"
        assert result["needs_alert"] is False

    @pytest.mark.asyncio
    async def test_parse_json_with_markdown_wrapper(self):
        """Should extract JSON from markdown code blocks."""
        analyzer = AIAnalyzer()
        response = '```json\n{"sentiment": "negative", "topic": "technical_issue", "confidence": "medium", "needs_alert": true}\n```'

        result = analyzer._parse_ai_response(response)

        assert result["sentiment"] == "negative"
        assert result["topic"] == "technical_issue"

    @pytest.mark.asyncio
    async def test_invalid_sentiment_defaults_to_neutral(self):
        """Invalid sentiment values should default to neutral."""
        analyzer = AIAnalyzer()
        response = '{"sentiment": "invalid_value", "topic": "general", "confidence": "low", "needs_alert": false}'

        result = analyzer._parse_ai_response(response)

        assert result["sentiment"] == "neutral"

    @pytest.mark.asyncio
    async def test_invalid_topic_defaults_to_general(self):
        """Invalid topic values should default to general."""
        analyzer = AIAnalyzer()
        response = '{"sentiment": "neutral", "topic": "nonexistent_topic", "confidence": "low", "needs_alert": false}'

        result = analyzer._parse_ai_response(response)

        assert result["topic"] == "general"

    @pytest.mark.asyncio
    async def test_missing_field_raises_error(self):
        """Missing required fields should raise ValueError."""
        analyzer = AIAnalyzer()
        response = '{"sentiment": "positive", "topic": "billing"}'

        with pytest.raises(ValueError, match="Missing required field"):
            analyzer._parse_ai_response(response)

    @pytest.mark.asyncio
    async def test_parse_json_with_extra_text(self):
        """Should extract JSON even with extra surrounding text."""
        analyzer = AIAnalyzer()
        response = 'Here is the analysis: {"sentiment": "positive", "topic": "general", "confidence": "high", "needs_alert": false} - done'

        result = analyzer._parse_ai_response(response)

        assert result["sentiment"] == "positive"
        assert result["topic"] == "general"

    @pytest.mark.asyncio
    async def test_invalid_confidence_defaults_to_medium(self):
        """Invalid confidence values should default to medium."""
        analyzer = AIAnalyzer()
        response = '{"sentiment": "neutral", "topic": "general", "confidence": "super_high", "needs_alert": false}'

        result = analyzer._parse_ai_response(response)

        assert result["confidence"] == "medium"


# ============================================================================
# UNIT TESTS - ML FALLBACK ANALYZER
# ============================================================================

class TestMLFallbackAnalyzer:
    """Test suite for ML-based fallback analyzer (RoBERTa)."""

    def test_negative_sentiment(self):
        """Test that negative feedback is correctly identified."""
        analyzer = MLFallbackAnalyzer()
        result = analyzer.analyze("This product is terrible and broken")
        assert result.sentiment == "negative"
        assert result.processing_method == "ml_fallback"
        assert result.confidence_score == "medium"

    def test_positive_sentiment(self):
        """Test that positive feedback is correctly identified."""
        analyzer = MLFallbackAnalyzer()
        result = analyzer.analyze("I absolutely love this product! It's amazing!")
        assert result.sentiment == "positive"
        assert result.processing_method == "ml_fallback"

    def test_alert_triggered_on_negative_sentiment(self):
        """Test that alerts are triggered for negative sentiment."""
        analyzer = MLFallbackAnalyzer()
        result = analyzer.analyze("I hate this service, it's awful")
        assert result.alert_triggered is True
        assert result.sentiment == "negative"

    def test_alert_triggered_on_mixed_sentiment(self):
        """Test that alerts are triggered for mixed sentiment."""
        analyzer = MLFallbackAnalyzer()
        result = analyzer.analyze("Love the features but hate the price")
        # ML model may classify this differently, accept negative or mixed
        assert result.sentiment in ["negative", "mixed"]
        assert result.alert_triggered is True

    def test_no_alert_on_positive_sentiment(self):
        """Test that positive feedback does not trigger alerts."""
        analyzer = MLFallbackAnalyzer()
        result = analyzer.analyze("This is wonderful, thank you!")
        assert result.alert_triggered is False
        assert result.sentiment == "positive"

    def test_topic_classification_billing(self):
        """Test that billing topics are correctly identified."""
        analyzer = MLFallbackAnalyzer()
        result = analyzer.analyze("I was charged twice for my subscription")
        assert result.topic == "billing"

    def test_topic_classification_technical(self):
        """Test that technical topics are correctly identified."""
        analyzer = MLFallbackAnalyzer()
        result = analyzer.analyze("The app keeps crashing when I click the button")
        assert result.topic == "technical_issue"

    def test_gibberish_input(self):
        """Test handling of gibberish/nonsensical input."""
        analyzer = MLFallbackAnalyzer()
        result = analyzer.analyze("asdfghjkl qwertyuiop")
        # ML model should handle this gracefully
        assert result.sentiment in ["positive", "neutral", "negative"]
        # Topic may be classified as anything based on token overlap
        assert result.topic in ["general", "product_features", "billing", "technical_issue", "customer_support", "delivery"]


# ============================================================================
# INTEGRATION TESTS - DATABASE
# ============================================================================

class TestDatabaseIntegration:
    """Test database operations."""

    @pytest.mark.asyncio
    async def test_save_feedback(self, db_session):
        """Test saving feedback to database."""
        result = AnalysisResult(
            sentiment="negative",
            topic="billing",
            confidence_score="high",
            alert_triggered=True,
            processing_method="ai"
        )

        feedback = await save_feedback(
            db_session,
            "Test feedback text",
            result
        )

        assert feedback.id is not None
        assert feedback.text == "Test feedback text"
        assert feedback.sentiment == "negative"
        assert feedback.topic == "billing"
        assert feedback.alert_triggered is True


# ============================================================================
# INTEGRATION TESTS - SENTIMENT-BASED ALERTS
# ============================================================================

class TestSentimentBasedAlerts:
    """Test that alerts are triggered based on sentiment, not keywords."""

    @pytest.mark.asyncio
    async def test_negative_sentiment_triggers_alert(self, db_session):
        """Test that negative sentiment triggers alert."""
        analyzer = MLFallbackAnalyzer()
        feedback_text = "This product is terrible and I hate it"

        result = analyzer.analyze(feedback_text)

        assert result.sentiment == "negative"
        assert result.alert_triggered is True

        # Save to database
        feedback = await save_feedback(db_session, feedback_text, result)
        assert feedback.alert_triggered is True

    @pytest.mark.asyncio
    async def test_positive_sentiment_no_alert(self, db_session):
        """Test that positive sentiment does not trigger alert."""
        analyzer = MLFallbackAnalyzer()
        feedback_text = "I love this product, it's amazing!"

        result = analyzer.analyze(feedback_text)

        assert result.sentiment == "positive"
        assert result.alert_triggered is False

    @pytest.mark.asyncio
    async def test_mixed_sentiment_triggers_alert(self, db_session):
        """Test that mixed sentiment triggers alert."""
        analyzer = MLFallbackAnalyzer()
        feedback_text = "Love the features but hate the billing"

        result = analyzer.analyze(feedback_text)

        # Mixed or negative both trigger alerts
        assert result.sentiment in ["negative", "mixed"]
        assert result.alert_triggered is True

    @pytest.mark.asyncio
    async def test_negative_without_keywords_triggers_alert(self, db_session):
        """Test that sentiment-based detection works without specific keywords."""
        analyzer = MLFallbackAnalyzer()
        feedback_text = "Disappointed with the service quality"

        result = analyzer.analyze(feedback_text)

        assert result.sentiment == "negative"
        assert result.alert_triggered is True


# ============================================================================
# INTEGRATION TESTS - SLACK WEBHOOK
# ============================================================================

class TestSlackWebhook:
    """Test Slack webhook integration."""

    @pytest.mark.asyncio
    async def test_slack_webhook_connection(self):
        """Test that Slack webhook can send messages."""
        if not config.ALERT_ENABLED or not config.ALERT_WEBHOOK_URL:
            pytest.skip("Slack alerts not configured")

        webhook_url = config.ALERT_WEBHOOK_URL

        payload = {
            "text": "🧪 Test message from unified test suite"
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(webhook_url, json=payload)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_alert_service_send_alert(self, db_session):
        """Test AlertService can send alerts to Slack."""
        if not config.ALERT_ENABLED:
            pytest.skip("Slack alerts not configured")

        alert_service = AlertService()
        analyzer = MLFallbackAnalyzer()

        feedback_text = "I want to cancel my subscription immediately!"
        result = analyzer.analyze(feedback_text)

        # Save feedback
        feedback = await save_feedback(db_session, feedback_text, result)

        # Send alert
        success = await alert_service.send_alert(
            feedback.id,
            feedback_text,
            result
        )

        assert success is True


# ============================================================================
# END-TO-END TESTS
# ============================================================================

class TestEndToEndWorkflow:
    """Test complete workflows from feedback to alert."""

    @pytest.mark.asyncio
    async def test_full_negative_feedback_flow(self, db_session):
        """Test complete flow: negative feedback → analysis → save → alert."""
        analyzer = MLFallbackAnalyzer()
        alert_service = AlertService()

        feedback_text = "This service is terrible and I'm very frustrated"

        # Step 1: Analyze
        result = analyzer.analyze(feedback_text)
        assert result.sentiment == "negative"
        assert result.alert_triggered is True

        # Step 2: Save to database
        feedback = await save_feedback(db_session, feedback_text, result)
        assert feedback.id is not None
        assert feedback.alert_triggered is True

        # Step 3: Send alert (if enabled)
        if config.ALERT_ENABLED:
            success = await alert_service.send_alert(
                feedback.id,
                feedback_text,
                result
            )
            assert success is True

    @pytest.mark.asyncio
    async def test_full_positive_feedback_flow(self, db_session):
        """Test complete flow: positive feedback → analysis → save → no alert."""
        analyzer = MLFallbackAnalyzer()

        feedback_text = "Excellent product, highly recommend!"

        # Step 1: Analyze
        result = analyzer.analyze(feedback_text)
        assert result.sentiment == "positive"
        assert result.alert_triggered is False

        # Step 2: Save to database
        feedback = await save_feedback(db_session, feedback_text, result)
        assert feedback.id is not None
        assert feedback.alert_triggered is False

    @pytest.mark.asyncio
    async def test_multiple_feedbacks_with_alerts(self, db_session):
        """Test processing multiple feedbacks with different sentiments."""
        analyzer = MLFallbackAnalyzer()
        alert_service = AlertService()

        test_cases = [
            {
                "text": "I want to cancel my subscription immediately!",
                "expected_sentiment": "negative",
                "expected_alert": True
            },
            {
                "text": "This is urgent - I need a refund right now!",
                "expected_sentiment": "negative",
                "expected_alert": True
            },
            {
                "text": "Great service, very happy!",
                "expected_sentiment": "positive",
                "expected_alert": False
            }
        ]

        for test in test_cases:
            result = analyzer.analyze(test["text"])

            # Verify sentiment
            assert result.sentiment == test["expected_sentiment"]
            assert result.alert_triggered == test["expected_alert"]

            # Save to database
            feedback = await save_feedback(db_session, test["text"], result)
            assert feedback.id is not None

            # Send alert if needed
            if result.alert_triggered and config.ALERT_ENABLED:
                success = await alert_service.send_alert(
                    feedback.id,
                    test["text"],
                    result
                )
                assert success is True


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
