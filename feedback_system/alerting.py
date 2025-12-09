"""Alerting system for urgent feedback."""
import logging
import httpx
from config import config
from schemas import AnalysisResult

logger = logging.getLogger(__name__)


class AlertService:
    """Service to send alerts for urgent feedback.

    Stubbed implementation per assignment requirements.
    In production, this would integrate with Slack, PagerDuty, etc.
    """

    def __init__(self):
        """Initialize alert service."""
        self.webhook_url = config.ALERT_WEBHOOK_URL
        self.enabled = config.ALERT_ENABLED

    async def send_alert(
        self,
        feedback_id: int,
        feedback_text: str,
        analysis: AnalysisResult
    ) -> bool:
        """Send alert for urgent feedback.

        Args:
            feedback_id: Database ID of the feedback
            feedback_text: Original feedback text
            analysis: Analysis result

        Returns:
            True if alert sent successfully, False otherwise
        """
        if not analysis.alert_triggered:
            return False

        if not self.enabled:
            logger.info(
                f"Alert would be sent for feedback {feedback_id} "
                f"(alerting disabled in config)"
            )
            return True

        alert_payload = self._build_alert_payload(
            feedback_id,
            feedback_text,
            analysis
        )

        try:
            if self.webhook_url:
                await self._send_webhook(alert_payload)
            else:
                # Log alert since webhook not configured
                logger.warning(
                    f"ALERT: Feedback #{feedback_id} requires attention - "
                    f"Sentiment: {analysis.sentiment}, Topic: {analysis.topic}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to send alert for feedback {feedback_id}: {e}")
            return False

    def _build_alert_payload(
        self,
        feedback_id: int,
        feedback_text: str,
        analysis: AnalysisResult
    ) -> dict:
        """Build alert payload for webhook.

        Args:
            feedback_id: Feedback ID
            feedback_text: Original text
            analysis: Analysis result

        Returns:
            Dictionary payload for webhook
        """
        # Slack-compatible format
        return {
            "text": f"🚨 Urgent Feedback Alert",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🚨 Urgent Customer Feedback"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Feedback ID:*\n{feedback_id}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Sentiment:*\n{analysis.sentiment}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Topic:*\n{analysis.topic}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Confidence:*\n{analysis.confidence_score}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Feedback:*\n{feedback_text[:500]}"
                    }
                }
            ]
        }

    async def _send_webhook(self, payload: dict) -> None:
        """Send webhook notification.

        Args:
            payload: JSON payload to send

        Raises:
            Exception: If webhook delivery fails
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                self.webhook_url,
                json=payload
            )
            response.raise_for_status()
            logger.info(f"Alert sent successfully to {self.webhook_url}")
