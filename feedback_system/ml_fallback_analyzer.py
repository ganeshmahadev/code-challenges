"""ML-based fallback analyzer using pre-trained RoBERTa sentiment model."""
import logging
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from schemas import AnalysisResult
from config import config

logger = logging.getLogger(__name__)


class MLFallbackAnalyzer:
    """ML-based fallback analyzer using sentiment-roberta-large-english.

    This provides higher accuracy fallback compared to keyword matching.
    Uses siebert/sentiment-roberta-large-english pre-trained model.
    """

    def __init__(self):
        """Initialize the ML fallback analyzer with pre-trained model."""
        try:
            logger.info("Loading RoBERTa sentiment model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "siebert/sentiment-roberta-large-english"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "siebert/sentiment-roberta-large-english"
            )
            self.model.eval()  # Set to evaluation mode

            # Topic keywords for topic classification
            self.topic_keywords = {
                "billing": [
                    "bill", "billing", "charge", "payment", "invoice", "refund",
                    "price", "cost", "subscription", "credit card", "pay"
                ],
                "technical_issue": [
                    "bug", "error", "crash", "broken", "not working", "loading",
                    "freeze", "slow", "lag", "glitch", "404", "500"
                ],
                "product_features": [
                    "feature", "functionality", "option", "setting", "ui", "ux",
                    "interface", "design", "layout", "button", "menu"
                ],
                "customer_support": [
                    "support", "help", "service", "representative", "agent",
                    "response", "reply", "contact", "chat", "email"
                ],
                "delivery": [
                    "delivery", "shipping", "ship", "arrived", "package", "order",
                    "tracking", "receive", "sent", "delivered"
                ]
            }

            self.available = True
            logger.info("RoBERTa model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.available = False

    def analyze(self, feedback_text: str) -> AnalysisResult:
        """Analyze feedback using RoBERTa sentiment model.

        Args:
            feedback_text: The customer feedback to analyze

        Returns:
            AnalysisResult with sentiment, topic, and alert status
        """
        if not self.available:
            # If ML model unavailable, return neutral sentiment
            logger.error("ML model unavailable - cannot analyze feedback")
            return AnalysisResult(
                sentiment="neutral",
                topic="general",
                confidence_score="low",
                alert_triggered=False,
                processing_method="ml_fallback"
            )

        # Analyze sentiment using RoBERTa
        sentiment = self._analyze_sentiment_ml(feedback_text)

        # Analyze topic using keyword matching (RoBERTa doesn't do topic classification)
        topic = self._analyze_topic(feedback_text.lower())

        # ML model provides medium confidence
        confidence_score = "medium"

        # Trigger alert based on sentiment AND confidence
        # Only alert on high/medium confidence to reduce noise from uncertain classifications
        alert_triggered = (
            sentiment in ["negative", "mixed"] and
            confidence_score in ["high", "medium"]
        )

        return AnalysisResult(
            sentiment=sentiment,
            topic=topic,
            confidence_score=confidence_score,
            alert_triggered=alert_triggered,
            processing_method="ml_fallback"
        )

    def _analyze_sentiment_ml(self, text: str) -> str:
        """Analyze sentiment using RoBERTa model.

        Args:
            text: Text to analyze

        Returns:
            Sentiment: positive, negative, or neutral
        """
        try:
            # Tokenize input (limit to 512 tokens for RoBERTa)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get predicted class
            # Model outputs: 0=NEGATIVE, 1=POSITIVE
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence_score = predictions[0][predicted_class].item()

            # Map to our sentiment categories
            if predicted_class == 1:
                sentiment = "positive"
            elif predicted_class == 0:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            # Check for mixed sentiment (low confidence on either class)
            if confidence_score < 0.65:
                sentiment = "mixed"

            logger.debug(f"ML Sentiment: {sentiment} (confidence: {confidence_score:.2f})")
            return sentiment

        except Exception as e:
            logger.error(f"ML sentiment analysis failed: {e}")
            # Default to neutral on error
            return "neutral"

    def _analyze_topic(self, text: str) -> str:
        """Determine primary topic based on keyword matching.

        Args:
            text: Lowercase text to analyze

        Returns:
            Topic category
        """
        topic_scores: Dict[str, int] = {}

        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                topic_scores[topic] = score

        if not topic_scores:
            return "general"

        # Return topic with highest score
        return max(topic_scores.items(), key=lambda x: x[1])[0]
