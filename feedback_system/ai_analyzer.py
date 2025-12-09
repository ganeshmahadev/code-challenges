"""AI-powered feedback analyzer using OpenAI."""
import json
import asyncio
from typing import Dict, Any
from openai import AsyncOpenAI
from config import config
from schemas import AnalysisResult


class AIAnalyzer:
    """Handles AI-based sentiment and topic analysis."""

    def __init__(self):
        """Initialize the AI analyzer."""
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
        self.model = config.AI_MODEL
        self.timeout = config.AI_TIMEOUT_SECONDS

    def _build_prompt(self, feedback_text: str) -> str:
        """Build the prompt for AI analysis.

        Design considerations:
        - Clear output format (JSON) for reliable parsing
        - Few-shot examples to improve accuracy
        - Explicit handling of edge cases (sarcasm, multiple topics)
        """
        topics_list = ", ".join(config.SUPPORTED_TOPICS)

        prompt = f"""Analyze the following customer feedback and return a JSON response.

FEEDBACK: "{feedback_text}"

Analyze this feedback and return ONLY a valid JSON object with these fields:
- sentiment: one of ["positive", "negative", "neutral", "mixed"]
- topic: one of [{topics_list}] (choose the PRIMARY topic if multiple exist)
- confidence: one of ["high", "medium", "low"] based on clarity of the feedback
- needs_alert: This is automatically determined by sentiment (negative or mixed = true)

Rules:
1. If sentiment is unclear or sarcastic, use "mixed" and set confidence to "low"
2. If feedback mentions multiple topics, pick the one that needs most attention
3. needs_alert is automatically set to true when sentiment is "negative" or "mixed" (indicates customer dissatisfaction)
4. For gibberish or non-feedback text, use: sentiment="neutral", topic="general", confidence="low"

Examples:
- "Love the product but billing is broken!" → {{"sentiment": "mixed", "topic": "billing", "confidence": "high", "needs_alert": true}}
- "Cancel my account immediately!" → {{"sentiment": "negative", "topic": "general", "confidence": "high", "needs_alert": true}}
- "This product is amazing!" → {{"sentiment": "positive", "topic": "product_features", "confidence": "high", "needs_alert": false}}
- "asdfghjkl" → {{"sentiment": "neutral", "topic": "general", "confidence": "low", "needs_alert": false}}

Return ONLY the JSON object, no additional text:"""

        return prompt

    async def analyze(self, feedback_text: str) -> AnalysisResult:
        """Analyze feedback using OpenAI API.

        Args:
            feedback_text: The customer feedback to analyze

        Returns:
            AnalysisResult with sentiment, topic, and alert status

        Raises:
            Exception: If AI provider fails (caller should handle with fallback)
        """
        if not self.client:
            raise Exception("OpenAI client not configured")

        prompt = self._build_prompt(feedback_text)

        try:
            # Use asyncio timeout to enforce response time constraint
            async with asyncio.timeout(self.timeout):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a customer feedback analyzer. Always respond with valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_tokens=150  # Keep it short for faster response
                )

            result_text = response.choices[0].message.content.strip()

            # Parse the JSON response
            result = self._parse_ai_response(result_text)

            # Determine alert based on sentiment AND confidence
            # Only alert on high/medium confidence to reduce noise
            alert_triggered = (
                result["sentiment"] in ["negative", "mixed"] and
                result["confidence"] in ["high", "medium"]
            )

            return AnalysisResult(
                sentiment=result["sentiment"],
                topic=result["topic"],
                confidence_score=result["confidence"],
                alert_triggered=alert_triggered,
                processing_method="ai"
            )

        except asyncio.TimeoutError:
            raise Exception(f"AI provider timeout after {self.timeout}s")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse AI response: {e}")
        except Exception as e:
            raise Exception(f"AI provider error: {str(e)}")

    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate AI response.

        Handles common AI output issues:
        - Extra text around JSON
        - Missing fields
        - Invalid enum values
        """
        # Try to extract JSON if there's extra text
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(response_text[start:end])
            else:
                raise

        # Validate required fields
        required_fields = ["sentiment", "topic", "confidence", "needs_alert"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Validate sentiment values
        valid_sentiments = ["positive", "negative", "neutral", "mixed"]
        if result["sentiment"] not in valid_sentiments:
            result["sentiment"] = "neutral"

        # Validate topic
        if result["topic"] not in config.SUPPORTED_TOPICS:
            result["topic"] = "general"

        # Validate confidence
        valid_confidence = ["high", "medium", "low"]
        if result["confidence"] not in valid_confidence:
            result["confidence"] = "medium"

        return result
